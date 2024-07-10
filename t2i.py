import torch
from diffusers import DiffusionPipeline
from utils import (
    attn_maps,
    cross_attn_init,
    register_cross_attention_hook,
    set_layer_with_name_and_path,
    save_by_timesteps_and_path,
    save_by_timesteps
)


import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--prompt', type=str, default="a sheep with a lion head", help='prompt for image generation')
    parser.add_argument('--height', type=int, default=512, help='height of image')
    parser.add_argument('--width', type=int, default=768, help='width of image')
    parser.add_argument('--model_name', type=str, default='sd3', help='model name')
    parser.add_argument('--cache_dir', type=str, default='/homes/55/runjia/scratch/diffusion_model_weights', help='cache directory')
    return parser.parse_args()



def main():
    ##### 1. Init modules #####
    cross_attn_init()
    ###########################
    args = parse_args()
    prompt = args.prompt
    height = args.height
    width = args.width
    model_name = args.model_name
    cache_dir = args.cache_dir

    if model_name == 'sd3':
        repo_id = 'stabilityai/stable-diffusion-3-medium-diffusers'
    elif model_name == 'sd2':
        repo_id = 'stabilityai/stable-diffusion-2-1'
    elif model_name == 'sdxl':
        repo_id = 'stabilityai/stable-diffusion-xl-base-1.0'
    pipe = DiffusionPipeline.from_pretrained(
        repo_id,
        torch_dtype=torch.float16,
        cache_dir=cache_dir
    )


    pipe = pipe.to("cuda:0")

    if model_name == 'sd3':
        pipe.transformer = set_layer_with_name_and_path(pipe.transformer)
        pipe.transformer = register_cross_attention_hook(pipe.transformer)
    else:
        ##### 2. Replace modules and Register hook #####
        pipe.unet = set_layer_with_name_and_path(pipe.unet)
        pipe.unet = register_cross_attention_hook(pipe.unet)
        ################################################

    image = pipe(
        prompt,
        height=height,
        width=width,
        num_inference_steps=15,
    ).images[0]
    image.save('test.png')

    ##### 3. Process and Save attention map #####
    print('resizing and saving ...')

    ##### 3-1. save by timesteps and path (2~3 minutes) #####
    # save_by_timesteps_and_path(pipe.tokenizer, prompt, height, width)
    #########################################################

    ##### 3-2. save by timesteps (1~2 minutes) #####
    save_by_timesteps(pipe.tokenizer, prompt, height, width)
    ################################################

if __name__ == '__main__':
    main()