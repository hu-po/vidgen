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
import logging

logging.basicConfig(level=logging.INFO)

def make_short(base_output_dir: str, output_video_filename: str, story_prompt: str, style_prompt: str):
    session_id = str(uuid.uuid4())[:6]
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
    logging.info(output)

    output = replicate.run(
        "tencent/hunyuan-video:847dfa8b01e739637fc76f480ede0c1d76408e1d694b830b5dfb8e547bf98405",
        input={
            "width": 854,
            "height": 480,
            "prompt": "A cat walks on the grass, realistic style.",
            "flow_shift": 7,
            "infer_steps": 50,
            "video_length": 129,
            "embedded_guidance_scale": 6
        }
    )
    logging.info(output)

    output = replicate.run(
        "pollinations/music-gen:9b8643c06debace10b9026f94dcb117f61dc1fee66558a09cde4cfbf51bcced6",
        input={
            "text": "A dynamic blend of hip-hop and orchestral elements, with sweeping strings and brass, evoking the vibrant energy of the city.",
            "duration": 12
        }
    )
    logging.info(output)

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