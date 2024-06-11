
from typing import List
from loguru import logger
import pathlib
import pyarrow
import os
import boto3
from botocore.exceptions import NoCredentialsError
import numpy as np

def get_s3_object_uris(
    s3_folder_uri: str,
    limit: int = None,
    use_pyarrow: bool = False,
    fs: pyarrow.fs.FileSystem = None,
) -> List[str]:
    # remove s3:// prefix
    prefix = "s3://"
    if s3_folder_uri.startswith(prefix):
        s3_folder_uri = s3_folder_uri[len(prefix) :]

    if use_pyarrow:
        if fs is None:
            raise ValueError("You need to provide a filesystem when using pyarrow")
        uris = fs.get_file_info(pyarrow.fs.FileSelector(s3_folder_uri, recursive=True))
        uris = [f"s3://{file_info.path}" for file_info in uris if file_info.is_file]
        if limit is not None:
            return uris[:limit]
        return uris

    my_session = boto3.session.Session()
    s3 = my_session.client("s3")

    # split bucket / folder
    bucket, folder = s3_folder_uri.split(".com/", 1)
    bucket = f"{bucket}.com"

    paginator = s3.get_paginator("list_objects_v2")
    page_iterator = paginator.paginate(Bucket=bucket, Prefix=folder)

    uris = []
    for page in page_iterator:
        if limit is not None and len(uris) - 1 > limit:
            break
        for obj in page.get("Contents", []):
            key = obj["Key"]
            # only if obj is file (not subfolder)
            if not key.endswith("/"):
                uris.append(f"s3://{bucket}/{obj['Key']}")

    if limit is not None:
        return uris[:limit]
    return uris

#%%
#  Copyright 2023 Canva Inc. All Rights Reserved

def configure_aws_assume_role_provider(model_trainer_role_arn: str) -> None:
    """
    Configure boto3 credentials with assume role provider
    (see https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html#assume-role-provider).

    How it works:

    Anyscale cluster role used by EC2 needs to assume model-trainer's role
    (in another account) to be able to run model training. To achieve this we rely
    on boto3's assume role provider feature. Thus, role assumption is performed on the application
    level. This function writes an aws config file set by the environment variable
    AWS_CONFIG_FILE. This variable is baked into the Anyscale base image and points
    to a directory in cluster storage, so that all nodes in the cluster can use it.

    Call this function in the very beginning of your training code.

    Arg:
        model_trainer_role_arn: The arn of the model trainer role.
    """
    if "AWS_CONFIG_FILE" not in os.environ:
        raise RuntimeError("AWS_CONFIG_FILE environment variable is not set")

    aws_config_file = pathlib.Path(os.getenv("AWS_CONFIG_FILE"))
    logger.info(f"AWS_CONFIG_FILE={aws_config_file}")

    parent_dir = pathlib.Path(aws_config_file.parents[0])
    logger.info(f"Creating dir {parent_dir} if it doesn't exist")
    parent_dir.mkdir(parents=True, exist_ok=True)

    if aws_config_file.exists():
        logger.info(f"aws config already exists: {aws_config_file.read_text()}")
    else:
        logger.info("Writing aws config")
        with open(aws_config_file, "w") as f:
            f.write("[profile default]\n")
            f.write(f"role_arn={model_trainer_role_arn}\n")
            f.write("credential_source=Ec2InstanceMetadata\n")
            f.write("region=us-east-1\n")


def upload_to_s3(file_name, bucket, object_name=None):
    """
    Upload a file to an S3 bucket
    
    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """
    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = file_name
    # Upload the file
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except FileNotFoundError:
        print("The file was not found")
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False
    return True