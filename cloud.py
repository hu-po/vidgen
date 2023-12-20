import base64
import glob
import os
import replicate
import subprocess
import uuid
import time
from io import BytesIO

import requests
from PIL import Image

def make_short(base_output_dir: str, output_video_filename: str, story_prompt: str, style_prompt: str):
    # Generate a unique id for this generation session
    session_id = str(uuid.uuid4())[:6]

    # Create a output folder for the session id and use that as the output dir
    output_dir = os.path.join(base_output_dir, session_id)
    os.makedirs(output_dir, exist_ok=True)


    output = replicate.run(
    "mistralai/mixtral-8x7b-instruct-v0.1:2b56576fcfbe32fa0526897d8385dd3fb3d36ba6fd0dbe033c72886b81ade93e",
    input={
        "top_k": 50,
        "top_p": 0.9,
        "prompt": "Write a sequence of short image prompts for the scenes in a movie trailer about horror movie about an evil banana. Each prompt= will be used as input to an image generation model. Each scene description uses adjectives and other prompt engineering tricks for high quality cinematic images. Separate each prompt with newlines, do not number the lines. There should be 10 prompts total.",
        "temperature": 0.6,
        "max_new_tokens": 1024,
        "prompt_template": "<s>[INST] {prompt} [/INST] ",
        "presence_penalty": 0,
        "frequency_penalty": 0
    }
    )
    print(output)


    output = replicate.run(
    "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
    input={
        "seed": 234234,
        "width": 544,
        "height": 960,
        "prompt": "3. A shot of a shadowy figure, hunched over and clutching the evil banana, its eyes glowing with a sinister light.",
        "refine": "expert_ensemble_refiner",
        "scheduler": "K_EULER",
        "lora_scale": 0.6,
        "num_outputs": 1,
        "guidance_scale": 7.5,
        "apply_watermark": False,
        "high_noise_frac": 0.8,
        "negative_prompt": "",
        "prompt_strength": 0.8,
        "num_inference_steps": 25
    }
    )
    print(output)


    output = replicate.run(
    "stability-ai/stable-video-diffusion:3f0457e4619daac51203dedb472816fd4af51f3149fa7a9e0b5ffcf1b8172438",
    input={
        "cond_aug": 0.02,
        "decoding_t": 7,
        "input_image": "https://example.com/out-0.png",
        "video_length": "14_frames_with_svd",
        "sizing_strategy": "maintain_aspect_ratio",
        "motion_bucket_id": 127,
        "frames_per_second": 6
    }
    )
    print(output)


    output = replicate.run(
    "pollinations/music-gen:9b8643c06debace10b9026f94dcb117f61dc1fee66558a09cde4cfbf51bcced6",
    input={
        "text": "A dynamic blend of hip-hop and orchestral elements, with sweeping strings and brass, evoking the vibrant energy of the city.",
        "duration": 12
    }
    )
    print(output)


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
