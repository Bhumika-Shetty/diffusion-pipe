# Summary

| Model          | LoRA | Full Fine Tune | fp8/quantization |
|----------------|------|----------------|------------------|
|SDXL            |✅    |✅              |❌                |
|Flux            |✅    |✅              |✅                |
|LTX-Video       |✅    |❌              |❌                |
|HunyuanVideo    |✅    |❌              |✅                |
|Cosmos          |✅    |❌              |❌                |
|Lumina Image 2.0|✅    |✅              |❌                |
|Wan2.1          |✅    |❌              |✅                |


## SDXL
```
[model]
type = 'sdxl'
checkpoint_path = '/data2/imagegen_models/sdxl/sd_xl_base_1.0_0.9vae.safetensors'
dtype = 'bfloat16'
# You can train v-prediction models (e.g. NoobAI vpred) by setting this option.
#v_pred = true
# Min SNR is supported. Same meaning as sd-scripts
#min_snr_gamma = 5
# Debiased estimation loss is supported. Same meaning as sd-scripts.
#debiased_estimation_loss = true
# You can set separate learning rates for unet and text encoders. If one of these isn't set, the optimizer learning rate will apply.
unet_lr = 4e-5
text_encoder_1_lr = 2e-5
text_encoder_2_lr = 2e-5
```
Unlike other models, for SDXL the text embeddings are not cached, and the text encoders are trained.

SDXL can be full fine tuned. Just remove the [adapter] table in the config file. You will need 48GB VRAM. 2x24GB GPUs works with pipeline_stages=2.

SDXL LoRAs are saved in Kohya sd-scripts format. SDXL full fine tune models are saved in the original SDXL checkpoint format.

## Flux
```
[model]
type = 'flux'
# Path to Huggingface Diffusers directory for Flux
diffusers_path = '/data2/imagegen_models/FLUX.1-dev'
# You can override the transformer from a BFL format checkpoint.
#transformer_path = '/data2/imagegen_models/flux-dev-single-files/consolidated_s6700-schnell.safetensors'
dtype = 'bfloat16'
# Flux supports fp8 for the transformer when training LoRA.
transformer_dtype = 'float8'
# Resolution-dependent timestep shift towards more noise. Same meaning as sd-scripts.
flux_shift = true
# For FLEX.1-alpha, you can bypass the guidance embedding which is the recommended way to train that model.
#bypass_guidance_embedding = true
```
For Flux, you can override the transformer weights by setting transformer_path to an original Black Forest Labs (BFL) format checkpoint. For example, the above config loads the model from Diffusers format FLUX.1-dev, but the transformer_path, if uncommented, loads the transformer from Flux Dev De-distill.

Flux LoRAs are saved in Diffusers format.

## LTX-Video
```
[model]
type = 'ltx-video'
diffusers_path = '/data2/imagegen_models/LTX-Video'
dtype = 'bfloat16'
timestep_sample_method = 'logit_normal'
```
LTX-Video LoRAs are saved in Diffusers format.

## HunyuanVideo
```
[model]
type = 'hunyuan-video'
# Can load Hunyuan Video entirely from the ckpt path set up for the official inference scripts.
#ckpt_path = '/home/anon/HunyuanVideo/ckpts'
# Or you can load it by pointing to all the ComfyUI files.
transformer_path = '/data2/imagegen_models/hunyuan_video_comfyui/hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors'
vae_path = '/data2/imagegen_models/hunyuan_video_comfyui/hunyuan_video_vae_bf16.safetensors'
llm_path = '/data2/imagegen_models/hunyuan_video_comfyui/llava-llama-3-8b-text-encoder-tokenizer'
clip_path = '/data2/imagegen_models/hunyuan_video_comfyui/clip-vit-large-patch14'
# Base dtype used for all models.
dtype = 'bfloat16'
# Hunyuan Video supports fp8 for the transformer when training LoRA.
transformer_dtype = 'float8'
# How to sample timesteps to train on. Can be logit_normal or uniform.
timestep_sample_method = 'logit_normal'
```
HunyuanVideo LoRAs are saved in a Diffusers-style format. The keys are named according to the original model, and prefixed with "transformer.". This format will directly work with ComfyUI.

## Cosmos
```
[model]
type = 'cosmos'
# Point these paths at the ComfyUI files.
transformer_path = '/data2/imagegen_models/cosmos/cosmos-1.0-diffusion-7b-text2world.pt'
vae_path = '/data2/imagegen_models/cosmos/cosmos_cv8x8x8_1.0.safetensors'
text_encoder_path = '/data2/imagegen_models/cosmos/oldt5_xxl_fp16.safetensors'
dtype = 'bfloat16'
```
Tentative support is added for Cosmos (text2world diffusion variants). Compared to HunyuanVideo, Cosmos is not good for fine-tuning on commodity hardware.

1. Cosmos supports a fixed, limited set of resolutions and frame lengths. Because of this, the 7b model is actually slower to train than HunyuanVideo (12b parameters), because you can't get away with training on lower-resolution images like you can with Hunyuan. And video training is nearly impossible unless you have enormous amounts of VRAM, because for videos you must use the full 121 frame length.
2. Cosmos seems much worse at generalizing from image-only training to video.
3. The Cosmos base model is much more limited in the types of content that it knows, which makes fine tuning for most concepts more difficult.

I will likely not be actively supporting Cosmos going forward. All the pieces are there, and if you really want to try training it you can. But don't expect me to spend time trying to fix things if something doesn't work right.

Cosmos LoRAs are saved in ComfyUI format.

## Lumina Image 2.0
```
[model]
type = 'lumina_2'
# Point these paths at the ComfyUI files.
transformer_path = '/data2/imagegen_models/lumina-2-single-files/lumina_2_model_bf16.safetensors'
llm_path = '/data2/imagegen_models/lumina-2-single-files/gemma_2_2b_fp16.safetensors'
vae_path = '/data2/imagegen_models/lumina-2-single-files/flux_vae.safetensors'
dtype = 'bfloat16'
lumina_shift = true
```
See the [Lumina 2 example dataset config](../examples/recommended_lumina_dataset_config.toml) which shows how to add a caption prefix and contains the recommended resolution settings.

In addition to LoRA, Lumina 2 supports full fine tuning. It can be fine tuned at 1024x1024 resolution on a single 24GB GPU. For FFT, delete or comment out the [adapter] block in the config. If doing FFT with 24GB VRAM, you will need to use an alternative optimizer to lower VRAM use:
```
[optimizer]
type = 'adamw8bitkahan'
lr = 5e-6
betas = [0.9, 0.99]
weight_decay = 0.01
eps = 1e-8
gradient_release = true
```

This uses a custom AdamW8bit optimizer with Kahan summation (required for proper bf16 training), and it enables an experimental gradient release for more VRAM saving. If you are training only at 512 resolution, you can remove the gradient release part. If you have a >24GB GPU, or multiple GPUs and use pipeline parallelism, you can perhaps just use the normal adamw_optimi optimizer type.

Lumina 2 LoRAs are saved in ComfyUI format.

## Wan2.1
```
[model]
type = 'wan'
ckpt_path = '/data2/imagegen_models/Wan2.1-T2V-1.3B'
dtype = 'bfloat16'
# You can use fp8 for the transformer when training LoRA.
#transformer_dtype = 'float8'
timestep_sample_method = 'logit_normal'
```

Wan2.1 t2v variants are supported. Set ckpt_path to the original model checkpoint directory, e.g. [Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B). The 1.3B is tested and confirmed working. I haven't tried the 14B yet. I don't know what the optimal training settings are.

Wan2.1 LoRAs are saved in ComfyUI format.