from huggingface_hub import HfApi
from pathlib import Path
import argparse
import tkinter as tk
import src.experiments.config as cfg

ORG_NAME = "gatech-deep-learning-project"

def huggingface_upload_model(local_path):
    local_path = Path(local_path)
    
    if not local_path.exists():
        print(f"Path does not exist: {local_path}")
        return
    
    api = HfApi()

    model_family = local_path.parent.name

    repo_id = f"{ORG_NAME}/{model_family}"
    
    print(f"Starting upload for {local_path.name}...")
    
    # validate model type
    valid_model_types = [experiment[0] for experiment in cfg.EXPERIMENTS.values()]
    if model_family not in valid_model_types:
        raise ValueError(f"Unknown model {valid_model_types}")
    
    # create repo if it does not exist
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        exist_ok=True,
        private=True
    )

    # upload file
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=local_path.name,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Uploaded {local_path.name} after training"
    )
    
    print("="*70)
    print(f"Uploaded {local_path.name} to {repo_id}")
    print("="*70)
    
def check_hf_login():
    try:
        api = HfApi()
        user_info = api.whoami()
        print(f"[HUGGINGFACE] Logged in as: {user_info["name"]}")
        return True
    except Exception:
        print("NOT LOGGED IN TO HUGGINGFACE")
        print("Please log into huggingface using 'hf auth login' in the command line with your api key")
        return False
        
    
if __name__=="__main__":
    hf_status = check_hf_login()
    
    if hf_status:
        parser = argparse.ArgumentParser(description='Upload Model to Huggingface')
        parser.add_argument("--model_path", type=str, required=True)
        
        args = parser.parse_args()
        
        huggingface_upload_model(args.model_path)
    