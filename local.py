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
docker run -p 5000:5000 --gpus=all sdxl
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


def make_docker(name: str):
    nuke_docker()
    docker_process = subprocess.Popen(
        ["docker", "run", "--rm", "-p", "5000:5000", "--gpus=all", name]
    )
    time.sleep(20)  # Let the docker container startup
    return docker_process


def extract_frame_number(frame_path):
    # Extract the number from the filename, assuming the filename format is controlnet_pose_<number>.png
    frame_name = frame_path.split("/")[-1]
    return int(frame_name.split("_")[-1].split(".")[0])


def make_short(base_output_dir: str, output_video_filename: str, story_prompt: str, style_prompt: str):
    # Generate a unique id for this generation session
    session_id = str(uuid.uuid4())[:6]

    # Create a output folder for the session id and use that as the output dir
    output_dir = os.path.join(base_output_dir, session_id)
    os.makedirs(output_dir, exist_ok=True)

    # ---- MIXTRAL
    docker_process = make_docker("mixtral")
    response = requests.post(
        "http://localhost:5000/predictions",
        headers={"Content-Type": "application/json"},
        json={
            "input": {
                "prompt": f"Write a sequence of short image prompts for the scenes in a movie trailer about {story_prompt}. Each prompt= will be used as input to an image generation model. Each scene description uses adjectives and other prompt engineering tricks for high quality cinematic images. Separate each prompt with newlines, do not number the lines. There should be 10 prompts total.",
                "top_k": 50,
                "top_p": 0.9,
                "temperature": 0.6,
                "max_new_tokens": 1024,
                "presence_penalty": 0,
                "frequency_penalty": 0,
            }
        },
    )
    scene_prompts = response.json()["output"].split("\n")
    docker_process.terminate()
    nuke_docker()
    # ---- SDXL
    docker_process = make_docker("sdxl")
    for i, scene_prompt in enumerate(scene_prompts):
        response = requests.post(
            "http://localhost:5000/predictions",
            headers={"Content-Type": "application/json"},
            json={
                "input": {
                    "width": 544,
                    "height": 960,
                    "prompt": f"{scene_prompt}, {style_prompt}",
                    "refine": "expert_ensemble_refiner",
                    "scheduler": "K_EULER",
                    "lora_scale": 0.6,
                    "num_outputs": 1,
                    "guidance_scale": 7.5,
                    "apply_watermark": False,
                    "high_noise_frac": 0.8,
                    "negative_prompt": "",
                    "prompt_strength": 0.8,
                    "num_inference_steps": 25,
                }
            },
        )
        img = Image.open(
            BytesIO(base64.b64decode(response.json()["output"].split(",")[1]))
        )
        img.save(os.path.join(output_dir, f"keyframe_{i:05}.png"))
    docker_process.terminate()
    nuke_docker()
    # ---- SVD
    docker_process = make_docker("svd")
    for i, keyframe_path in enumerate(
        sorted(
            glob.glob(os.path.join(output_dir, "keyframe_*.png")),
            key=extract_frame_number,
        )
    ):
        with open(keyframe_path, "rb") as img_file:
            response = requests.post(
                "http://localhost:5000/predictions",
                headers={"Content-Type": "application/json"},
                json={
                    "input": {
                        "input_image": f"data:image/png;base64,{base64.b64encode(img_file.read()).decode('utf-8')}",
                        "cond_aug": 0.02,
                        "decoding_t": 7,
                        "video_length": "14_frames_with_svd",
                        "sizing_strategy": "maintain_aspect_ratio",
                        "motion_bucket_id": 20,
                        "frames_per_second": 6,
                    },
                },
            )
            

    # ---- MUSICGEN
    docker_process = make_docker("musicgen_container")
    response = requests.post(
        "http://localhost:5000/predictions",
        headers={"Content-Type": "application/json"},
        json={
            "input": {
                "text": f"cool trendy bass heavy electronic music that would fit nicely in a movie trailer about {prompt}",
                "duration": 12,
            }
        },
    )

    # Combine scenes and audio with ffmpeg
    os.system(
        f"ffmpeg -r 6 -f image2 -s 768x768 -i {os.path.join(output_dir, 'keyframe_%05d.png')} -i {os.path.join(output_dir, 'audio.wav')} -shortest -c:v libx264 -pix_fmt yuv420p {os.path.join(output_dir, output_video_filename)}"
    )


if __name__ == "__main__":
    style_prompts = [
        "cyberpunk art, inspired by Victor Mosquera, conceptual art, style of raymond swanland, yume nikki, restrained, ghost in the shell",
        "inspired by Krenz Cushart, neoism, kawacy, wlop, gits anime",
    ]
    story_prompts = [
        "cute story about a penguin and a polar bear",
        "horror movie about an evil banana",
    ]
    for story_prompt in story_prompts:
        for style_prompt in style_prompts:
            make_short(
                "/home/oop/dev/data/",
                f"out_{story_prompt[:5]}_{style_prompt[:5]}.mp4",
                story_prompt,
                style_prompt,
            )
