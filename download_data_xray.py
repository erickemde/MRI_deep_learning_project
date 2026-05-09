# Must create kaggle.json file in
# - Windows: C:/Users/<Windows-username>/.kaggle/kaggle.json
# - Mac/Linux: ~/.kaggle/kaggle.json

import kaggle
import os

# Make directory for raw data
os.makedirs("data_raw_xray", exist_ok=True)

# Kaggle dataset path
dataset_id = "paultimothymooney/chest-xray-pneumonia"

# Download using kaggle api
kaggle.api.dataset_download_files(dataset_id, path="data_raw_xray", unzip=True)
