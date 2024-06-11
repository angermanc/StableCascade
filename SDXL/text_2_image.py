#%%
import torch
import wandb
import PIL
from diffusers import (DiffusionPipeline, 
                       EulerDiscreteScheduler, 
                       AutoencoderKL,
                       DDIMScheduler)
from diffusers import image_processor
import time

# initialize a wandb run
wandb.init(project="chris-playground", job_type="SDXL")


#%%
# define experiment configs
config = wandb.config
config.stable_diffusion_checkpoint = "stabilityai/stable-diffusion-xl-base-1.0"
config.refiner_checkpoint = "stabilityai/stable-diffusion-xl-refiner-1.0"
config.offload_to_cpu = False
config.compile_model = True
config.prompt_1 = "a painting of a man standing on a street corner"
config.prompt_2 = config.prompt_1
config.negative_prompt_1 = ""
config.negative_prompt_2 = config.negative_prompt_1
config.num_images_per_prompt = 16
config.seed = 42
config.use_ensemble_of_experts = True
config.num_inference_steps = 50
config.num_refinement_steps = 50
config.high_noise_fraction = 0.8
# config.scheduler_kwargs = {
#     "beta_end": 0.012,
#     "beta_schedule": "scaled_linear", # one of ["linear", "scaled_linear"]
#     "beta_start": 0.00085,
#     "interpolation_type": "linear", # one of ["linear", "log_linear"]
#     "num_train_timesteps": 1000,
#     "prediction_type": "epsilon", # one of ["epsilon", "sample", "v_prediction"]
#     "steps_offset": 1,
#     "timestep_spacing": "leading", # one of ["linspace", "leading"]
#     "trained_betas": None,
#     "use_karras_sigmas": False,
# }




#%%
# Define base model
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
inference_scheduler = DDIMScheduler.from_pretrained(
    config.stable_diffusion_checkpoint,
    subfolder='scheduler')

pipe = DiffusionPipeline.from_pretrained(
    config.stable_diffusion_checkpoint,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
    vae = vae,
    scheduler=inference_scheduler
)


# Offload to CPU in case of OOM
if config.offload_to_cpu:
    pipe.enable_model_cpu_offload()
else:
    pipe.to("cuda")
    
# Compile model using `torch.compile`,
# this might give a significant speedup
if config.compile_model:
    pipe.unet = torch.compile(
        pipe.unet, mode="reduce-overhead", fullgraph=True
    )

#%%
# Define base model
refiner = DiffusionPipeline.from_pretrained(
    config.refiner_checkpoint,
    text_encoder_2=pipe.text_encoder_2,
    vae=pipe.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)

# Offload to CPU in case of OOM
if config.offload_to_cpu:
    refiner.enable_model_cpu_offload()
else:
    refiner.to("cuda")

# Compile model using `torch.compile`,
# this might give a significant speedup
if config.compile_model:
    refiner.unet = torch.compile(
        refiner.unet, mode="reduce-overhead", fullgraph=True
    )



#%%

# making the experiments deterministic
if config.seed is not None:
    torch.manual_seed(config.seed)

if config.use_ensemble_of_experts:
    for k in range(4):
        start_time = time.time()
        latent = pipe(
            prompt=config.prompt_1,
            prompt_2=config.prompt_2, # equals prompt if not specified
            negative_prompt=config.negative_prompt_1,
            negative_prompt_2=config.negative_prompt_2, # equals negative_prompt if not specified
            output_type="latent",
            num_inference_steps=config.num_inference_steps,
            denoising_end=config.high_noise_fraction,
            num_images_per_prompt = config.num_images_per_prompt,
        )
        

        '''
        scaling_factor 0.18215
        This is used to scale the latent space to have unit variance when training 
        the diffusion model. The latents are scaled with the formula z = z * scaling_factor
        before being passed to the diffusion model. When decoding, the latents are scaled 
        back to the original scale with the formula: z = 1 / scaling_factor * z. 
        '''
        # Postprocess the latents from the base model
        # by passing it though the base model's VAE
        # with torch.no_grad():         
        #     image = pipe.vae.decode(latent.images * (1/0.18215)).sample     

        # image = (image / 2 + 0.5).clamp(0, 1)     
        # image = image.detach().cpu().permute(0, 2, 3, 1).numpy()      
        # image = (image * 255).round().astype("uint8")     
        # unrefined_image = PIL.Image.fromarray(image[0]) 
        
        refined_image = refiner(
            prompt=config.prompt_1,
            prompt_2=config.prompt_2,
            negative_prompt=config.negative_prompt_1,
            negative_prompt_2=config.negative_prompt_2,
            image=latent.images,
            num_inference_steps=config.num_refinement_steps,
            denoising_start=config.high_noise_fraction,
            num_images_per_prompt = config.num_images_per_prompt
        ).images

        print("Inference time: %.1f" %(time.time()-start_time))


#%%
import numpy as np
list = [np.array(img) for img in refined_image]
new_img = np.concatenate(list, axis=1)
new_img = PIL.Image.fromarray(new_img)

new_img.save(fp = "/home/ray/image-generation/output/sdxl/%d.jpg" %time.time())

#%%
# using base image samples by base model and give to refiner -> improve high-quality results
if not config.use_ensemble_of_experts:
    latent = pipe(
        prompt=config.prompt_1,
        prompt_2=config.prompt_2,
        negative_prompt=config.negative_prompt_1,
        negative_prompt_2=config.negative_prompt_2,
        output_type="latent",
        num_inference_steps=config.num_inference_steps,
    )

    # Postprocess the latents from the base model
    # by passing it though the base model's VAE
    with torch.no_grad():         
        image = pipe.vae.decode(latent.images * (1/0.18215)).sample     
    image = (image / 2 + 0.5).clamp(0, 1)     
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()      
    image = (image * 255).round().astype("uint8")     
    unrefined_image = PIL.Image.fromarray(image[0]) 
    
    
    refined_image = refiner(
        prompt=config.prompt_1,
        prompt_2=config.prompt_2,
        negative_prompt=config.negative_prompt_1,
        negative_prompt_2=config.negative_prompt_2,
        image=latent.images,
        num_inference_steps=config.num_refinement_steps,
    ).images[0]



#%%
# # Create a [wandb table](https://docs.wandb.ai/guides/tables) 
# table = wandb.Table(columns=[
#     "Prompt-1", "Prompt-2", "Negative-Prompt-1", "Negative-Prompt-2",
#     "Unrefined-Image", 
#     "Refined-Image", "Use-Ensemble-of-Experts",
# ])

# refined_image = wandb.Image(refined_image)
# unrefined_image = wandb.Image(unrefined_image)

# # Add the images to the table
# table.add_data(
#     config.prompt_1, config.prompt_2,
#     config.negative_prompt_1, config.negative_prompt_2,
#     unrefined_image, 
#     refined_image, config.use_ensemble_of_experts,
# )

# # Log the images and table to wandb
# wandb.log({
#     "Unrefined-Image": unrefined_image,
#     "Refined-Image": refined_image,
#     "Text-to-Image": table
# })

# #%%
# # finish the experiment
# wandb.finish()
# # %%
