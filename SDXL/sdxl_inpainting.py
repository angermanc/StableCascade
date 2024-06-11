#%%
import os
import torch
import wandb
import PIL
from diffusers import DiffusionPipeline, EulerDiscreteScheduler, AutoencoderKL
from diffusers import AutoPipelineForInpainting, AutoPipelineForText2Image
from diffusers.utils import load_image, make_image_grid
import time

checkpoint = "stabilityai/stable-diffusion-xl-base-1.0"
#%%
# Define base model
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
inference_scheduler = EulerDiscreteScheduler.from_pretrained(
    checkpoint,
    subfolder='scheduler')

pipe_text2image = DiffusionPipeline.from_pretrained(
    checkpoint,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
    vae = vae,
    scheduler=inference_scheduler
)

pipe = AutoPipelineForInpainting.from_pipe(pipe_text2image).to('cuda')

#%%
sample = '/home/ray/image-generation/inpainting_data/jeans_and_shorts'

prompt = sample.split('/')[-1]
prompt = prompt.replace('_',' ')

img_data = [f for f in os.listdir(sample) if f.lower().endswith('.jpg')]

img_url = os.path.join(sample, img_data[0])
mask_url = os.path.join(sample, 'mask.png')
# img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl-text2img.png"
# mask_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/sdxl-inpaint-mask.png"
init_image = load_image(img_url)
mask_image = load_image(mask_url)

#%%
import numpy as np
def largest_centered_square_crop(pil_image):
    image = pil_image.convert('L')  # Convert to grayscale
    image_array = np.array(image)/255.
    # Identify positions of the object (pixels with value 1)
    object_pixels = np.argwhere(image_array == 1)
    if object_pixels.size == 0:
        raise ValueError("No object pixels found in the image.")
    
    # Find the bounding box of the object
    min_row, min_col = object_pixels.min(axis=0)
    max_row, max_col = object_pixels.max(axis=0)

    # Calculate the center of the bounding box
    center_row = (min_row + max_row) // 2
    center_col = (min_col + max_col) // 2

    # Determine the original image dimensions
    h, w = image_array.shape

    # Determine the size of the largest possible square that can be taken from the image
    square_size = min(h, w)

    # Calculate the coordinates for the square crop to center around the bounding box center
    half_size = square_size // 2
    left = max(0, center_col - half_size)
    top = max(0, center_row - half_size)
    right = min(w, center_col + half_size)
    bottom = min(h, center_row + half_size)

    # Adjust if the crop extends beyond the image boundaries
    if right - left < square_size:
        if left == 0:
            right = square_size
        elif right == w:
            left = w - square_size
    if bottom - top < square_size:
        if top == 0:
            bottom = square_size
        elif bottom == h:
            top = h - square_size
    
    return left, top, right, bottom

def replace_region(original_image, replacement_image, left, top, right, bottom):
    # Calculate the size of the region to be replaced
    region_width = right - left
    region_height = bottom - top
    # Resize the replacement image to fit within the specified coordinates
    resized_replacement_image = replacement_image.resize((region_width, region_height))
    # Paste the resized replacement image into the original image at the specified coordinates
    original_image.paste(resized_replacement_image, (left, top))
    return original_image

    
coordinates = largest_centered_square_crop(mask_image)
cropped_mask = mask_image.crop(coordinates)
cropped_init = init_image.crop(coordinates)

#%%
for seed in [0, 42]:
    generator = torch.Generator(device='cuda')
    generator.manual_seed(seed)
    image = pipe(
                prompt=prompt, 
                image=cropped_init, 
                mask_image=cropped_mask, 
                strength=0.85, 
                guidance_scale=12.5,
                generator = generator).images[0]


    new_image = replace_region(init_image.copy(),image, *coordinates)
    grid = make_image_grid([init_image, mask_image, new_image], rows=1, cols=3)
    new_image.save(fp = "/home/ray/image-generation/output/sdxl/%d.jpg" %time.time())
    time.sleep(2)
    grid.save(fp = "/home/ray/image-generation/output/sdxl/%d.jpg" %time.time())
#%%
