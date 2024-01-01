# vidgen
generate shorts

### ComfyUI

Kinda completely unrelated to this project, but pasting here for easy reference

image

```
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI/
conda create -n comfy python=3.12
conda activate comfy
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
pip install -r requirements.txt
wget -c https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors -P ./models/checkpoints/
wget -c https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0.safetensors -P ./models/checkpoints/
wget -c https://huggingface.co/comfyanonymous/clip_vision_g/resolve/main/clip_vision_g.safetensors -P ./models/clip_vision/
wget -c https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors -P ./models/vae/
python main.py --force-fp16
```

video

```
wget -c https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/resolve/main/svd_xt.safetensors -P ./models/checkpoints/
wget -c https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/resolve/main/svd_xt_image_decoder.safetensors ./models/checkpoints/
wget -c https://huggingface.co/stabilityai/stable-video-diffusion-img2vid/resolve/main/svd.safetensors -P ./models/checkpoints/
wget -c https://huggingface.co/stabilityai/stable-video-diffusion-img2vid/resolve/main/svd_image_decoder.safetensors ./models/checkpoints/
wget -O models/checkpoints/workflow_image_to_video.json https://comfyanonymous.github.io/ComfyUI_examples/video/workflow_image_to_video.json
wget -O models/checkpoints/workflow_txt_to_img_to_video.json https://comfyanonymous.github.io/ComfyUI_examples/video/workflow_txt_to_img_to_video.json
python main.py --force-fp16
```
