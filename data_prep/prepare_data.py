#%%
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
    upload_to_s3,
    configure_ray_for_safer_s3_reading
)
import shutil
import subprocess


#%%
@ray.remote
def save_image_to_jpg(item, index, path):
    try:
        image_data = io.BytesIO(item)
        image = Image.open(image_data)
        #absolute path necessary, otherwise no data is saved
        path = "%s/%06d.jpg" %(path,index)
        image.save(path, format='JPEG')
        return path
    except Exception as e:
        print(f"Error occurred while saving image {index}: {e}")
        return None
    
@ray.remote
def save_caption_to_txt(item, index, path):
    try:
        path = "%s/%06d.txt" %(path,index)
        with open(path, "w") as file:
            file.write(item)
        return path
    except Exception as e:
        print(f"Error occurred while saving image {index}: {e}")
        return None
    

#%%

fs = pyarrow.fs.S3FileSystem(role_arn=config.MODEL_TRAINER_ARN, region=config.AWS_REGION)
configure_aws_assume_role_provider(config.MODEL_TRAINER_ARN)
session = boto3.session.Session()
s3_uri = "s3://ingredient-generation-model-trainer.canva.com/image-generation/canva-internal/processed/sd/res512-20m-cogvlm-description-gptv-jpg-only-0X/"
uris = sorted(get_s3_object_uris(s3_uri))
K=8000

while K < len(uris):

    dir_path = '/mnt/local_storage/data_%d'%K
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)  # Remove the directory and its contents
    os.makedirs(dir_path) 
     
    configure_ray_for_safer_s3_reading()

    ds = ray.data.read_parquet(
        uris[K:(K+2000)],
        shuffle=None,
        columns = ['cogvlm_captions','image_bytes'],
        filesystem=fs
    )
    
    index = 0
    for batch in ds.iter_batches(batch_size=1_000, batch_format='pandas'):
        image_bytes_list = batch['image_bytes'].tolist()
        caption_list = batch['cogvlm_captions'].tolist()

        # image data
        futures = [save_image_to_jpg.remote(
            image_bytes, idx + index, dir_path) for idx, image_bytes in enumerate(image_bytes_list)]
        ray.get(futures)

        # text data
        futures_txt = [save_caption_to_txt.remote(
            captions, idx + index, dir_path) for idx, captions in enumerate(caption_list)]
        ray.get(futures_txt)

        index += len(image_bytes_list)

    K += 2000


    # Define the command and arguments
    command = [
        'tar',
        '--sort=name',
        '-cf',
        '/mnt/cluster_storage/dataset_0X_%d.tar' %int(K/2000),
        dir_path
    ]

    # Run the command
    result = subprocess.run(command, check=True)

    command2 = [
        "rm",
        "-r",
        dir_path
    ]
    # Run the command
    result2 = subprocess.run(command2, check=True)


# %%


configure_aws_assume_role_provider(config.MODEL_TRAINER_ARN)

tar_files = sorted([os.path.join('/mnt/cluster_storage', f) for f in os.listdir('/mnt/cluster_storage') if 'tar' in f])
bucket_name = 'ingredient-generation-model-trainer.canva.com'


# Example usage
for f in tar_files:
    object_name = 'image-generation/test/chris/cascade-data/%s' %f.split('/')[-1] 
    upload_successful = upload_to_s3(f, bucket_name, object_name)
    if upload_successful:
        print("Upload successful")
    else:
        print("Upload failed")



# %%
