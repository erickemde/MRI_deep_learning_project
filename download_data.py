import kaggle
import os

os.makedirs("data_raw", exist_ok=True)

dataset_id = "masoudnickparvar/brain-tumor-mri-dataset"

kaggle.api.dataset_download_files(dataset_id, path="data_raw", unzip=True)