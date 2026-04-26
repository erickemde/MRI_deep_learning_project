# deep_learning_project
Project Repo for CS7643 (Deep Learning)

# Documentation
## Connect to PACE-ICE (Optional)
1. Follow instructions [PACE-ICE GPU Resources](https://edstem.org/us/courses/89061/discussion/7602224)
2. Navigate to the "~/scratch" directory
3. Transfer necessary files (drag and drop with VS Code "Remote-SSH" extension)
    - environment.yaml
    - download_data.py
    - data_preprocessing.py
    - train_model.py
    - gradcam_inference.py
    - src/
    - configs/
4. Load anaconda
    - "module spider anaconda"
    - "module load \<anaconda version\>"
5. Create kaggle directory and store "kaggle.json" file (see Data Setup #2)
    - In remote terminal: "mkdir -p ~/.kaggle"
    - In local terminal: "scp C:\Users\YourName\.kaggle\kaggle.json YourEmail@login-ice.pace.gatech.edu:~/.kaggle/kaggle.json"
    - Update permissions in remote terminal: "chmod 600 ~/.kaggle/kaggle.json"
6. Weights and Biases API
    - Create an account at [W&B](wandb.ai)
    - In User Settings, create an API key
    - In ICE-PACE, enter command "wandb login" and paste API key
7. Slurm batch file
    - Create an sbatch file using "template.sbatch". Here is a helpful resource [Using Slurm on ICE](https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042096)
    - Create an output directory if needed (e.g. /slurm_out/)
    - Execute "sbatch YourFilename.sbatch"

## Data Setup
1. Create an environment with required packages
    - "conda env create -f environment.yaml" OR
    - "pip install -r requirements.txt"
2. Execute "python download_data.py"
    - Kaggle dataset ID: "masoudnickparvar/brain-tumor-mri-dataset"
    - Must create a kaggle account and a Legacy API Key (in Settings)
    - Store the kaggle.json file
        - Windows - C:/Users/\<Windows-username\>/.kaggle/kaggle.json
        - Mac/Linux - ~/.kaggle/kaggle.json
    - Finally, run "python download_data.py" and the data will populate in "/data_raw/"
3. Execute "python data_processing.py"
    - De-duplicates images via perceptual hashing
    - Creates train/val/test (69.6%/7.7%/22.7%) sets in "/data/"

## Train a model
Basic usage: "python train_model.py --config configs/baseline.yaml"
- Dependencies:
    - src/data/dataset
    - src/data/augmentation
    - src/models/lit_vgg_attention
    - src/models/lit_agg
    - src/visualization/gradcam
    - src/experiments/config
1. Set up or select a config file in "/configs/"
    - For new experiments, define the experiment/parameters in "/src/experiments/config.py"
2. Execute "python train_model.py --config configs/YourConfig.yaml"
    - Sets up experiment
    - Loads data
        - Augmentation (details in src/data/augmentation.py) uses torchvision.transforms.v2.Compose with transformations (if use_augmentation=True):
            - Resize to 224x224
            - RandomHorizontalFlip
            - RandomAffine
            - Normalize
    - Builds model by calling lit_vgg_attention.VGGLightningWrapper
        - Cross-entropy loss
        - Adam optimizer
        - ReduceLROnPlateau scheduler
        - Current model options (classes in lig_vgg_attention.py):
            - VGG19Baseline
            - VGG19SEAttention
            - VGG19SoftmaxAttention
        - All models built from class src.models.vgg19_attention.VGG19Model
            - Architecture: torchvision.models.vgg19_bn
            - Pretrained weights (if pretrained==True): VGG19_BN_Weights.IMAGENET1K_V1
            - To unfreeze weights: set unfreeze_from_layer in "/src/experiments/config.py"
    - Trains model using pytorch_lightning.Trainer
    - Generates GradCAM visualizations with class src.visualization.gradcam.GradCAM
        - Registers hooks
        - Generates GradCAM heatmap
        - Overlays on original image
        - Generates examples for each class and misclassified examples

### GradCAM
Basic usage: "python gradcam_inference.py --config configs/baseline.yaml"
- Same as "train_model.py", but loads pretrained model from checkpoint and performs GradCAM only
- Dependencies:
    - src/data/dataset
    - src/data/augmentation
    - src/models/lit_vgg_attention
    - src/models/lit_agg
    - src/visualization/gradcam
    - src/experiments/config

### Ablation Study
Basic usage: "python ablation_study.py --config configs/ablation.yaml"
- Same as "train_model.py", but will run all EXPERIMENTS (in src/experiments/config.py) if passed "experiment: ablation"
- Additional features:
    - "generate_gradcam: True" to generate GradCAM images for all models (False or omit to skip)
    - "evaluate: best | last" to evaluate the best model checkpoints or final models against the test dataset ("no" or omit to skip)