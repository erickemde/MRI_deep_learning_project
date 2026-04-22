#!/bin/bash
#SBATCH -J YourJobName              # Job name  
#SBATCH -N1 --ntasks-per-node=1          # Number of nodes and cores per node required 
#SBATCH --gres=gpu:H100:1                # GPU type (H100) and number of GPUs 
#SBATCH --mem-per-gpu=12GB              # Memory per CPU core, 8 CPUs/GPU 
#SBATCH -t 00:30:00                        # Duration of the job (Ex: 1 hour) 
#SBATCH -o /home/hice1/YourUsername/scratch/deep_learning_project/slurm_out/slurm_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL       # Mail preferences 
#SBATCH --mail-user=YourEmail@gatech.edu # E-mail address for notifications

# Set working directory
cd /home/hice1/YourUsername/scratch/deep_learning_project

# Load modules
module load anaconda3
module load cuda

# Activate env
source $(conda info --base)/etc/profile.d/conda.sh
conda activate CS7643_deep_learning_project

# Execute script
python train_model.py --config configs/baseline.yaml