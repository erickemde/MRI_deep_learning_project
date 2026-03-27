import kaggle
import os

os.makedirs("data", exist_ok=True)

dataset_id = "masoudnickparvar/brain-tumor-mri-dataset"

kaggle.api.dataset_download_files(dataset_id, path="./data", unzip=True)