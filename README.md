# deep_learning_project
Project Repo for CS7643 (Deep Learning)

# Connect to PACE-ICE (Optional)
1. Follow instructions [PACE-ICE GPU Resources](https://edstem.org/us/courses/89061/discussion/7602224)
2. Navigate to the "~/scratch" directory
3. Transfer necessary files (drag and drop with VS Code "Remote-SSH" extension)
    - environment.yaml
    - download_data.py
    - data_preprocessing.py
4. Load anaconda
    - "module spider anaconda"
    - "module load \<anaconda version\>"
5. Create kaggle directory and store "kaggle.json" file (see Data Setup #2)
    - In remote terminal: "mkdir -p ~/.kaggle"
    - In local terminal: "scp C:\Users\YourName\.kaggle\kaggle.json YourEmail@login-ice.pace.gatech.edu:~/.kaggle/kaggle.json"
    - Update permissions in remote terminal: "chmod 600 ~/.kaggle/kaggle.json"

# Data Setup
1. Create an environment with required packages
    - "conda env create -f environment.yaml" OR
    - "pip install -r requirements.txt"
2. Execute "python download_data.py"
    - Kaggle dataset ID: "masoudnickparvar/brain-tumor-mri-dataset"
    - Must create a kaggle account and a Legacy API Key (in Settings)
    - Store the kaggle.json file
        - Windows - C:\Users\<Windows-username>\.kaggle\kaggle.json
        - Mac/Linux - ~/.kaggle/kaggle.json
    - Finally, run "python download_data.py" and the data will populate in "/data_raw/"
3. Execute "python data_processing.py"
    - De-duplicates images via perceptual hashing
    - Creates train/val/test (69.6%/7.7%/22.7%) sets in "/data/"

# Train a model
Basic usage: "python train_model.py --config configs/baseline.yaml"

## GradCAM
Basic usage: "python gradcam_inference.py --config configs/baseline.yaml"