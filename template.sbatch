#!/bin/bash
#SBATCH -J YourJobName              # Job name  
#SBATCH -N1 --ntasks-per-node=1          # Number of nodes and cores per node required 
#SBATCH --gres=gpu:H100:1                # GPU type (H100) and number of GPUs 
#SBATCH --mem-per-gpu=12GB              # Memory per CPU core, 8 CPUs/GPU 
#SBATCH -t 00:30:00                        # Duration of the job (Ex: 1 hour) 
#SBATCH -o /home/hice1/YourUsername/scratch/deep_learning_project/slurm_out/slurm_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL       # Mail preferences 
#SBATCH --mail-user=YourEmail@gatech.edu # E-mail address for notifications

# Redirect cache dirs to scratch so they don't fill home quota
export WANDB_DIR=/home/hice1/YourUsername/scratch/wandb
export WANDB_CACHE_DIR=/home/hice1/YourUsername/scratch/wandb_cache
export PIP_CACHE_DIR=/home/hice1/YourUsername/scratch/pip_cache
export TMPDIR=/home/hice1/YourUsername/scratch/tmp

# Set working directory
cd /home/hice1/YourUsername/scratch/deep_learning_project

# Load modules
module load anaconda3
module load cuda

# Execute script
/home/hice1/YourUsername/scratch/envs/CS7643_deep_learning_project/bin/python ablation_study.py --config configs/ablation.yaml