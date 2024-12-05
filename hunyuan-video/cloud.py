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
import asyncio
import csv
import re
import json
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable, Dict, List, Optional
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .ai import AI_MODEL_MAP, ENABLED_MODELS

log = logging.getLogger(__name__)
log.info(f"Enabled models: {ENABLED_MODELS}")

NUM_SCENES : int = 3
WIDTH : int = 854
HEIGHT : int = 480

async def async_generate_storyboard(ai_models: List[str], story_prompt: str, style_prompt: str) -> Dict[str, str]:
    log.debug(f"Starting AI inference and storyboard generation - models: {ai_models}")
    try:
        if not ENABLED_MODELS:
            log.error("No AI APIs enabled")
            raise ValueError("No AI APIs enabled")
        
        prompt = f"Generate a storyboard for a short video with {NUM_SCENES} scenes. The story prompt is: {story_prompt}. The style prompt is: {style_prompt}. Return the storyboard as a newline separated list"
        tasks = []
        task_keys = []
        
        for ai_model in ai_models:
            if ai_model not in ENABLED_MODELS:
                log.error(f"Requested model {ai_model} not in enabled models: {ENABLED_MODELS}")
                raise ValueError(f"Model {ai_model} not enabled")
            tasks.append(await AI_MODEL_MAP[ai_model](prompt))
            task_keys.append(ai_model)
        
        if not tasks:
            log.error("No tasks created - check enabled models and analyses")
            raise ValueError("No enabled models found in model map")
        
        log.debug(f"Executing {len(tasks)} inference tasks")
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        results = {
            key: resp if not isinstance(resp, Exception) else str(resp)
            for key, resp in zip(task_keys, responses)
        }
        
        log.debug("Inference results:")
        for key, result in results.items():
            if isinstance(result, Exception):
                log.error(f"{key} failed: {str(result)}")
            else:
                log.debug(f"{key} succeeded")
        
        return results
        
    except Exception as e:
        log.error(f"AI inference error: {str(e)}", exc_info=True)
        return {"error": f"AI inference failed: {str(e)}"}


def make_short(base_output_dir: str, story_prompt: str, style_prompt: str):
    session_id = str(uuid.uuid4())[:6]
    output = asyncio.run(async_generate_storyboard(ENABLED_MODELS, story_prompt, style_prompt))
    logging.info(output)

    for ai_model, scenes in output.items():
        model_output_dir = os.path.join(base_output_dir, session_id, ai_model)
        os.makedirs(model_output_dir, exist_ok=True)

        with open(os.path.join(model_output_dir, 'prompt.txt'), 'w') as f:
            f.write(f"story_prompt:\n{story_prompt}\n")
            f.write(f"style_prompt:\n{style_prompt}\n")

        scene_videos = []
        for i, scene in enumerate(scenes.split('\n')):
            video_path = os.path.join(model_output_dir, f"scene_{i}.mp4")
            replicate.run(
                "tencent/hunyuan-video:847dfa8b01e739637fc76f480ede0c1d76408e1d694b830b5dfb8e547bf98405",
                input={
                    "width": WIDTH,
                    "height": HEIGHT,
                    "prompt": scene,
                    "flow_shift": 7,
                    "infer_steps": 50,
                    "video_length": 129,
                    "embedded_guidance_scale": 6
                },
                output_file=video_path
            )
            scene_videos.append(video_path)
            logging.info(f"Generated video for scene {i} using {ai_model}: {video_path}")

        combined_video_path = os.path.join(model_output_dir, f"final_video_{session_id}.mp4")
        ffmpeg_command = [
            "ffmpeg",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", os.path.join(model_output_dir, "filelist.txt"),
            "-c", "copy",
            combined_video_path
        ]

        subprocess.run(ffmpeg_command, check=True)
        logging.info(f"Combined video saved to: {combined_video_path}")

        final_video_path = os.path.join(model_output_dir, f"final_video_{session_id}_{ai_model}.mp4")
        os.rename(combined_video_path, final_video_path)
        logging.info(f"Final video saved to: {final_video_path}")

        filelist_path = os.path.join(model_output_dir, "filelist.txt")
        with open(filelist_path, 'w') as f:
            for video in scene_videos:
                f.write(f"file '{video}'\n")


if __name__ == "__main__":
    style_prompts = [
        "realistic, cinematic, christopher nolan",
        "cyberpunk, arcane, pixar",
        "trippy, wes anderson",
    ]
    story_prompts = [
        "battle between the empires of Sam Altman and Elon Musk",
        "humanity progressing towards a utopian future",
    ]
    for story_prompt in story_prompts:
        for style_prompt in style_prompts:
            make_short("/home/oop/dev/data/hunyuan-video-replicate", story_prompt, style_prompt, ENABLED_MODELS)