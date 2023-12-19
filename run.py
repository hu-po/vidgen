"""
conda create -n vid2vid python=3.9
conda activate vid2vid
pip install replicate

https://replicate.com/stability-ai/stable-video-diffusion
docker run --name svd r8.im/stability-ai/stable-video-diffusion@sha256:3f0457e4619daac51203dedb472816fd4af51f3149fa7a9e0b5ffcf1b8172438
docker commit svd svd
"input": {
    "cond_aug": 0.02,
    "decoding_t": 7,
    "input_image": "https://replicate.delivery/pbxt/JvLi9smWKKDfQpylBYosqQRfPKZPntuAziesp0VuPjidq61n/rocket.png",
    "video_length": "14_frames_with_svd",
    "sizing_strategy": "maintain_aspect_ratio",
    "motion_bucket_id": 127,
    "frames_per_second": 6
}

https://replicate.com/stability-ai/sdxl
docker run --name sdxl r8.im/stability-ai/sdxl@sha256:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b
docker commit sdxl sdxl
"input": {
    "width": 768,
    "height": 768,
    "prompt": "An astronaut riding a rainbow unicorn, cinematic, dramatic",
    "refine": "expert_ensemble_refiner",
    "scheduler": "K_EULER",
    "lora_scale": 0.6,
    "num_outputs": 1,
    "guidance_scale": 7.5,
    "apply_watermark": false,
    "high_noise_frac": 0.8,
    "negative_prompt": "",
    "prompt_strength": 0.8,
    "num_inference_steps": 25
}

https://replicate.com/pollinations/music-gen
docker run --name musicgen_container r8.im/pollinations/music-gen@sha256:9b8643c06debace10b9026f94dcb117f61dc1fee66558a09cde4cfbf51bcced6
docker commit musicgen_container musicgen_container
"input": {
    "text": "cool trendy bass heavy electronic music that would fit nicely in an epic movie trailer",
    "duration": 12
}

https://replicate.com/mistralai/mixtral-8x7b-instruct-v0.1
docker run --name mixtral r8.im/mistralai/mixtral-8x7b-instruct-v0.1@sha256:2b56576fcfbe32fa0526897d8385dd3fb3d36ba6fd0dbe033c72886b81ade93e
docker commit mixtral mixtral
"input": {
    "top_k": 50,
    "top_p": 0.9,
    "prompt": "Write a bedtime story about neural networks I can read to my toddler",
    "temperature": 0.6,
    "max_new_tokens": 1024,
    "prompt_template": "<s>[INST] {prompt} [/INST] ",
    "presence_penalty": 0,
    "frequency_penalty": 0
}
"""

import base64
import glob
import os
import subprocess
import uuid
import time
from io import BytesIO

import requests
from PIL import Image


def nuke_docker():
    containers = os.popen("docker ps -aq").read().strip()
    if containers:
        os.system(f"docker kill {containers}")
        os.system(f"docker stop {containers}")
        os.system(f"docker rm {containers}")
    os.system("docker container prune -f")


def crop_center_square(input_video_path):
    # Get video details using ffprobe
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate",
        "-of",
        "csv=p=0",
        input_video_path,
    ]
    output = subprocess.check_output(cmd).decode("utf-8").strip().split(",")
    width, height, framerate = [int(output[0]), int(output[1]), output[2].split("/")[0]]
    min_dim = min(width, height)

    # Calculate the offset for cropping
    x_offset = (width - min_dim) // 2
    y_offset = (height - min_dim) // 2

    # Set output file name
    output_video_path = f"test.square.30fps.{min_dim}x{min_dim}.mp4"

    # Use ffmpeg to crop the video
    cmd = [
        "ffmpeg",
        "-i",
        input_video_path,
        "-vf",
        f"crop={min_dim}:{min_dim}:{x_offset}:{y_offset}",
        "-c:v",
        "libx264",
        "-r",
        "30",
        output_video_path,
    ]
    subprocess.call(cmd)

    print(f"Video cropped and saved to: {output_video_path}")


def extract_frame_number(frame_path):
    # Extract the number from the filename, assuming the filename format is controlnet_pose_<number>.png
    frame_name = frame_path.split("/")[-1]
    return int(frame_name.split("_")[-1].split(".")[0])


def process_video_frames(
    input_video_path: str = "/home/oop/dev/simicam/data/test.square.30fps.1080x1080.mp4",
    base_output_dir: str = "/home/oop/dev/simicam/logs/",
    output_video_filename: str = "out.mp4",
    prompt: str = "humanoid robot dancing test, unreal engine, 8k render",
    fps: int = 30,
    docker_url: str = "http://localhost:5000/predictions",
    **kwargs,
):
    # Generate a unique id for this generation session
    session_id = str(uuid.uuid4())[:6]

    # Create a output folder for the session id and use that as the output dir
    output_dir = os.path.join(base_output_dir, session_id)
    os.makedirs(output_dir, exist_ok=True)

    # Extract frames from video using ffmpeg
    os.system(f"ffmpeg -i {input_video_path} -vf fps={fps} {output_dir}/raw_%05d.png")

    # Run the controlnet docker container and remove it after use
    nuke_docker()
    docker_process = subprocess.Popen(
        [
            "docker",
            "run",
            "--rm",
            "-p",
            "5000:5000",
            "--gpus=all",
            "controlnet_container",
        ]
    )
    time.sleep(30)  # Let the docker container startup

    # Feed each frame to the controlnet docker container and save the output image
    for i, frame_path in enumerate(
        sorted(
            glob.glob(os.path.join(output_dir, "raw_*.png")), key=extract_frame_number
        )
    ):
        with open(frame_path, "rb") as img_file:
            response = requests.post(
                docker_url,
                headers={"Content-Type": "application/json"},
                json={
                    "input": {
                        "image": f"data:image/png;base64,{base64.b64encode(img_file.read()).decode('utf-8')}",
                        "prompt": prompt,
                        "seed": 42,
                    },
                },
            )
        img = Image.open(
            BytesIO(base64.b64decode(response.json()["output"][0].split(",")[1]))
        )
        img.save(os.path.join(output_dir, f"controlnet_pose_{i:05}.png"))
        img = Image.open(
            BytesIO(base64.b64decode(response.json()["output"][1].split(",")[1]))
        )
        img.save(os.path.join(output_dir, f"controlnet_full_{i:05}.png"))

    # Kill controlnet docker container
    docker_process.terminate()

    # Run the bgremoval docker container
    nuke_docker()
    docker_process = subprocess.Popen(
        [
            "docker",
            "run",
            "--rm",
            "-p",
            "5000:5000",
            "--gpus=all",
            "bgremoval_container",
        ]
    )
    time.sleep(20)  # Let the docker container startup

    # Feed each frame to the bgremoval docker container
    for i, frame_path in enumerate(
        sorted(
            glob.glob(os.path.join(output_dir, "controlnet_full_*.png")),
            key=extract_frame_number,
        )
    ):
        with open(frame_path, "rb") as img_file:
            response = requests.post(
                docker_url,
                headers={"Content-Type": "application/json"},
                json={
                    "input": {
                        "image": f"data:image/png;base64,{base64.b64encode(img_file.read()).decode('utf-8')}",
                    },
                },
            )
        img = Image.open(
            BytesIO(base64.b64decode(response.json()["output"].split(",")[1]))
        )
        img.save(os.path.join(output_dir, f"nobg_{i:05}.png"))

    # Kill controlnet docker container
    docker_process.terminate()
    os.system("docker kill $(docker ps -q)")

    # Combine frames into video
    os.system(
        f"ffmpeg -framerate {fps} -i {output_dir}/nobg_%05d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p {base_output_dir}/{output_video_filename}"
    )


if __name__ == "__main__":
    # Define the values you want to try for fps and prompt
    fps_values = [60]
    prompt_values = [
        # "robot, battle droid, unreal engine",
        # "shaman world of warcraft, mmo",
        "anime mma fighter",
        "captain america in style of moebius",
        "obama white suit",
        "elon musk anime",
        "elon musk pixar",
        "elon musk moebius",
    ]

    # Iterate over all combinations of fps and prompt values
    for fps in fps_values:
        for prompt in prompt_values:
            process_video_frames(
                input_video_path="/home/oop/dev/data/test.square.30fps.1080x1080.mp4",
                base_output_dir="/home/oop/dev/data/",
                output_video_filename=f"output_{fps}_{prompt[:5]}.mp4",
                prompt=prompt,
                fps=fps,
                docker_url="http://localhost:5000/predictions",
            )

    # Call the function with your video path
    # crop_center_square("/home/oop/Downloads/PXL_20231019_175956589.TS.mp4")
