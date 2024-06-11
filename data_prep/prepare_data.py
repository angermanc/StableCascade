#%%
!pip install loguru
import pyarrow.fs
from common import config
import ray
import io
from PIL import Image
import boto3
import boto3
import os
from utils import (
    configure_aws_assume_role_provider, 
    get_s3_object_uris,
    upload_to_s3
)
import shutil


#%%
K = 7
fs = pyarrow.fs.S3FileSystem(role_arn=config.MODEL_TRAINER_ARN, region=config.AWS_REGION)
configure_aws_assume_role_provider(config.MODEL_TRAINER_ARN)
session = boto3.session.Session()
s3_uri = "s3://ingredient-generation-model-trainer.canva.com/image-generation/canva-internal/processed/sd/FIXED-res512-20m-cogvlm-description-gptv-jpg-only-2X/"
uris = sorted(get_s3_object_uris(s3_uri))

ds = ray.data.read_parquet(
    uris[K*1000:(K+1)*1000],
    shuffle=None,
    columns = ['cogvlm_captions','image_bytes'],
    filesystem=fs
)



#%%
@ray.remote
def save_image_to_jpg(item, index):
    try:
        image_data = io.BytesIO(item)
        image = Image.open(image_data)
        #absolute path necessary, otherwise no data is saved
        path = "/mnt/local_storage/data/%06d.jpg" %index
        image.save(path, format='JPEG')
        return path
    except Exception as e:
        print(f"Error occurred while saving image {index}: {e}")
        return None
    
@ray.remote
def save_caption_to_txt(item, index):
    try:
        path = "/mnt/local_storage/data/%06d.txt" %index
        with open(path, "w") as file:
            file.write(item)
        return path
    except Exception as e:
        print(f"Error occurred while saving image {index}: {e}")
        return None
    
#%%
dir_path = '/mnt/local_storage/data'
if os.path.exists(dir_path):
    shutil.rmtree(dir_path)  # Remove the directory and its contents
os.makedirs(dir_path) 
index = 0
for batch in ds.iter_batches(batch_size=1000, batch_format='pandas'):
    image_bytes_list = batch['image_bytes'].tolist()
    caption_list = batch['cogvlm_captions'].tolist()

    # image data
    futures = [save_image_to_jpg.remote(
        image_bytes, idx + index) for idx, image_bytes in enumerate(image_bytes_list)]
    ray.get(futures)

    # text data
    futures_txt = [save_caption_to_txt.remote(
        captions, idx + index) for idx, captions in enumerate(caption_list)]
    ray.get(futures_txt)

    index += len(image_bytes_list)


# %%


configure_aws_assume_role_provider(config.MODEL_TRAINER_ARN)

# Example usage
file_name = '/mnt/local_storage/dataset_2X_%d.tar'%K
bucket_name = 'ingredient-generation-model-trainer.canva.com'
object_name = 'image-generation/test/chris/cascade-data/dataset_2X_%d.tar'%K  # Optional: if you want a different name in S3

upload_successful = upload_to_s3(file_name, bucket_name, object_name)
if upload_successful:
    print("Upload successful")
else:
    print("Upload failed")

# %%
