#%%
import os
import yaml
import torch
from tqdm import tqdm

os.chdir('..')
from inference.utils import *
from core.utils import load_or_fail
from train import WurstCoreC, WurstCoreB

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#%%
# SETUP STAGE C
config_file = 'configs/inference/stage_c_3b.yaml'
with open(config_file, "r", encoding="utf-8") as file:
    loaded_config = yaml.safe_load(file)

core = WurstCoreC(config_dict=loaded_config, device=device, training=False)

# SETUP STAGE B
config_file_b = 'configs/inference/stage_b_3b.yaml'
with open(config_file_b, "r", encoding="utf-8") as file:
    config_file_b = yaml.safe_load(file)
    
core_b = WurstCoreB(config_dict=config_file_b, device=device, training=False)

#%%
# SETUP MODELS & DATA
extras_c = core.setup_extras_pre()
models = core.setup_models(extras_c)
models.generator.eval().requires_grad_(False)
print("STAGE C READY")

extras_b = core_b.setup_extras_pre()
models_b = core_b.setup_models(extras_b, skip_clip=True)
models_b = WurstCoreB.Models(
   **{**models_b.to_dict(), 'tokenizer': models.tokenizer, 'text_model': models.text_model}
)
models_b.generator.bfloat16().eval().requires_grad_(False)
print("STAGE B READY")

#%%
# models = WurstCoreC.Models(
#    **{**models.to_dict(), 'generator': torch.compile(models.generator, mode="reduce-overhead", fullgraph=True)}
# )

# models_b = WurstCoreB.Models(
#    **{**models_b.to_dict(), 'generator': torch.compile(models_b.generator, mode="reduce-overhead", fullgraph=True)}
# )

#%%
batch_size = 4
# caption = "Cinematic photo of an anthropomorphic nerdy rodent sitting in a cafe reading a book"
captions = [
    # "a small house",
    "a man with wild hair looking into a crystal ball",
    # (
    #     "A plush monkey fording the Charles River on a log while wearing a "
    #     "Boston Red Sox hat with MIT in the background."
    # ),
    # "a painting of a man standing on a street corner",
    # "Beach house overlooking turquoise water at sunset"
]
height, width = 384, 384
stage_c_latent_shape, stage_b_latent_shape = calculate_latent_sizes(height, width, batch_size=batch_size)

# Stage C Parameters
extras_c.sampling_configs['cfg'] = 4
extras_c.sampling_configs['shift'] = 2
extras_c.sampling_configs['timesteps'] = 20
extras_c.sampling_configs['t_start'] = 1.0

# Stage B Parameters
extras_b.sampling_configs['cfg'] = 1.1
extras_b.sampling_configs['shift'] = 1
extras_b.sampling_configs['timesteps'] = 10
extras_b.sampling_configs['t_start'] = 1.0



#%%
import numpy as np
import time
array_list = []

# PREPARE CONDITIONS
for caption in captions:
    batch = {'captions': [caption] * batch_size}
    start_time = time.time()
    conditions = core.get_conditions(batch, models, extras_c, is_eval=True, is_unconditional=False, eval_image_embeds=False)
    unconditions = core.get_conditions(batch, models, extras_c, is_eval=True, is_unconditional=True, eval_image_embeds=False)    
    conditions_b = core_b.get_conditions(batch, models_b, extras_b, is_eval=True, is_unconditional=False)
    unconditions_b = core_b.get_conditions(batch, models_b, extras_b, is_eval=True, is_unconditional=True)

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        # torch.manual_seed(42)

        sampling_c = extras_c.gdf.sample(
            models.generator, conditions, stage_c_latent_shape,
            unconditions, device=device, **extras_c.sampling_configs,
        )
        for (sampled_c, _, _) in tqdm(sampling_c, total=extras_c.sampling_configs['timesteps']):
            sampled_c = sampled_c
            
        preview_c = models.previewer(sampled_c).float()
        show_images(preview_c)

        conditions_b['effnet'] = sampled_c
        unconditions_b['effnet'] = torch.zeros_like(sampled_c)

        sampling_b = extras_b.gdf.sample(
            models_b.generator, conditions_b, stage_b_latent_shape,
            unconditions_b, device=device, **extras_b.sampling_configs
        )
        for (sampled_b, _, _) in tqdm(sampling_b, total=extras_b.sampling_configs['timesteps']):
            sampled_b = sampled_b
        sampled = models_b.stage_a.decode(sampled_b).float()
    
    print('Inference time %.1f' %(time.time()-start_time))
    images = show_images(sampled, return_images = True)
    images.save(fp = "/home/ray/image-generation/StableCascade/output/t2i/%d.jpg" %time.time())
    array_list.append(np.array(images))
# %%
