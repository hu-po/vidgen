import os
import replicate
import uuid
import logging
import asyncio
import requests
from typing import Any, Dict, List
from ai import AI_MODEL_MAP, ENABLED_MODELS

log = logging.getLogger(__name__)
log.info(f"Enabled models: {ENABLED_MODELS}")

NUM_SCENES : int = 3
WIDTH : int = 480
HEIGHT : int = 854

async def async_generate_storyboard(story_prompt: str, style_prompt: str) -> Dict[str, str]:
    log.debug(f"Starting storyboard generation")
    try:
        if not ENABLED_MODELS:
            log.error("No AI APIs enabled")
            raise ValueError("No AI APIs enabled")
        
        prompt = f"""Generate video generation prompts for {NUM_SCENES} scenes that will be combined into a short video for a youtube short story.
The description of the story is: {story_prompt}.
In the folowing style: {style_prompt}.
Return the {NUM_SCENES} prompts in a newline separated list. ONLY return the prompts, nothing else."""
        tasks = []
        ai_models = []

#         example outputs
        
# "A sweeping panoramic shot of two vast futuristic cities, one dominated by sleek Musk-designed structures and the other by the organic, rounded architecture of Altman's empire, their skylines bristling with advanced technologies as they face each other across a vast expanse.\n\nAn intense close-up of Sam Altman and Elon Musk locked in a fierce mental duel, their eyes burning with determination as holographic displays and streams of data swirl around them, representing the clash of their artificial intelligences.\n\nA breathtaking aerial view of colossal robotic armies clashing in a scorched battleground between the two cities, their advanced weaponry and sleek designs contrasting with explosions and debris, as the fate of technological supremacy hangs in the balance."

# "A sweeping aerial shot revealing two vast, futuristic armies clashing amidst a desolate, rocky landscape.  One army utilizes sleek, chrome technology, the other, rugged, bio-mechanical weaponry.  The style is stark, realistic, and evokes the scale of a major war.  Muted color palette, emphasizing grays, browns, and metallics.\n\n\nClose-up on Sam Altman, commanding his forces from a high-tech mobile command center. He appears calm but calculating, his face etched with determination amidst the chaos of the battle. The scene should be claustrophobic and intense, contrasting the vastness of the war with his personal struggle.\n\n\nElon Musk, surrounded by his elite guard, leads a desperate counter-attack, utilizing a powerful, experimental weapon. The scene should be filled with explosive action, showcasing the raw power of the weapon and Musk's relentless, almost desperate, ambition.  The visual style should be gritty and visceral.\n"

        for ai_model in ENABLED_MODELS:
            tasks.append(AI_MODEL_MAP[ai_model](prompt))
            ai_models.append(ai_model)
        
        if not tasks:
            log.error("No tasks created - check enabled models and analyses")
            raise ValueError("No enabled models found in model map")
        
        log.debug(f"Executing {len(tasks)} inference tasks")
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        results = {
            key: resp if not isinstance(resp, Exception) else str(resp)
            for key, resp in zip(ai_models, responses)
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
    output = asyncio.run(async_generate_storyboard(story_prompt, style_prompt))
    logging.info(output)

    for ai_model, scenes in output.items():
        model_output_dir = os.path.join(base_output_dir, session_id, ai_model)
        os.makedirs(model_output_dir, exist_ok=True)

        with open(os.path.join(model_output_dir, 'prompt.txt'), 'w') as f:
            f.write(f"story_prompt:\n{story_prompt}\n")
            f.write(f"style_prompt:\n{style_prompt}\n")

        scene_videos = []
        scenes = [scene.strip() for scene in scenes.splitlines() if scene]
        for i, scene in enumerate(scenes):
            with open(os.path.join(model_output_dir, f"scene_{i}.txt"), 'w') as f:
                f.write(scene)
            output = replicate.run(
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
            )
            video_url = output  # Assuming output is the URL
            video_path = os.path.join(model_output_dir, f"scene_{i}.mp4")
            response = requests.get(video_url, stream=True)
            if response.status_code == 200:
                with open(video_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                logging.info(f"Downloaded video for scene {i} using {ai_model}: {video_path}")
            else:
                logging.error(f"Failed to download video for scene {i}: {response.status_code}")

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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("hunyuan-video.ai").setLevel(logging.DEBUG)
    logging.getLogger("hunyuan-video.cloud").setLevel(logging.DEBUG)

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
            make_short("/home/oop/dev/data/hunyuan-video-replicate", story_prompt, style_prompt)