from pathlib import Path


WANDB_ENTITY_NAME: str = "canva"
MODEL_NAME: str = "simple-model"

# The project name here should be the same as Anyscale project name
PROJECT_NAME: str = "image-generation"

ANYSCALE_NFS_CLUSTER_STORAGE: str = "/mnt/cluster_storage"
RAY_RESULTS_FOLDER: str = "ray_results"

CHECKPOINT_DIR: str = "checkpoints"

# For example, design-analysis-model-trainer, review-model-trainer etc.
MODEL_TRAINER_ARN: str = "arn:aws:iam::051687089423:role/service.ingredient-generation-model-trainer"

# For example, /review/model_trainer/wandb-password etc.
WANDB_SECRET_NAME: str = "/ingredient-generation/model-trainer/wandb-password"

AWS_REGION: str = "us-east-1"
TRANSIENT_BUCKET_NAME = "ingredient-generation-img-gen.canva.com"
PERMANENT_BUCKET_NAME = "ingredient-generation-model-trainer.canva.com"
ALL_CANVA_20M_PARQUET_URIS = f"s3://{TRANSIENT_BUCKET_NAME}/canva-internal-20M/all_parquet_uris_fixed_v2.txt"
ALL_CANVA_20M_IMGS_URIS = f"s3://{TRANSIENT_BUCKET_NAME}/canva-internal-20M-processed/metadata/all_images_uris/"

WORKING_DIR: str = str(Path(__file__).parent.parent.resolve())