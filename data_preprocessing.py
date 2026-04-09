"""
Data Preprocessing Pipeline for Brain Tumor Classification
Based on Aiya et al. (2025) with Enhanced Quality Assurance

Key Features:
1. Perceptual hashing (pHash) for duplicate detection
2. Separate handling of Training and Testing directories
3. Training → Train/Val split (90/10) per baseline methodology
4. Testing → Duplicate removal only (no split)

References:
- Zauner, C. (2010). Implementation and benchmarking of perceptual image hash functions
- Venkatesan et al. (2000). Robust image hashing
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import imagehash


def get_perceptual_hash(image_path, hash_size=8):
    """
    Perceptual Hash (pHash) based duplicate detection
    
    This method extracts low-frequency DCT coefficients from 32×32 
    grayscale-converted images to generate robust 64-bit image fingerprints.
    
    Args:
        image_path: Image file path
        hash_size: Hash size (8=64bit, 16=256bit)
    
    Returns:
        str: Image hash value (hexadecimal string)
    
    Reference:
        Zauner (2010): DCT-based perceptual hashing maintains stability 
        under content-preserving transformations including brightness/contrast 
        adjustments, JPEG compression artifacts, and minor geometric variations.
    """
    img = Image.open(image_path)
    img_hash = imagehash.phash(img, hash_size=hash_size)
    return str(img_hash)


def find_and_remove_duplicates(data_dir, remove=True, verbose=True):
    """
    Duplicate image detection and removal using perceptual hashing
    
    Args:
        data_dir: Directory to check (Training or Testing)
        remove: True=remove duplicates, False=detect only
        verbose: Print detailed output
    
    Returns:
        dict: {class_name: {
            'initial': int,
            'duplicates_found': int,
            'duplicates_removed': int,
            'remaining': int
        }}
    """
    print(f"\n{'='*60}")
    print(f"🔍 Duplicate Image Detection: {os.path.basename(data_dir)}")
    print(f"{'='*60}")
    print(f"   Method: Perceptual Hash (pHash)")
    print(f"   References: Zauner (2010), Venkatesan et al. (2000)\n")
    
    stats = {}
    total_duplicates = 0
    total_removed = 0
    
    # Process each class
    for class_name in sorted(os.listdir(data_dir)):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        
        print(f"   📁 {class_name}")
        
        # List image files
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        initial_count = len(image_files)
        
        # Hash computation and duplicate detection
        hash_dict = {}
        duplicates = []
        
        for img_file in image_files:
            img_path = os.path.join(class_path, img_file)
            
            try:
                img_hash = get_perceptual_hash(img_path)
                
                if img_hash in hash_dict:
                    # Duplicate found!
                    duplicates.append((img_file, hash_dict[img_hash]))
                    if verbose:
                        print(f"      ⚠️  Duplicate: {img_file} ≈ {hash_dict[img_hash]}")
                    
                    # Remove duplicate (optional)
                    if remove:
                        os.remove(img_path)
                        total_removed += 1
                else:
                    hash_dict[img_hash] = img_file
                    
            except Exception as e:
                print(f"      ❌ Error: {img_file} - {e}")
        
        # Statistics
        remaining = len([f for f in os.listdir(class_path) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        stats[class_name] = {
            'initial': initial_count,
            'duplicates_found': len(duplicates),
            'duplicates_removed': len(duplicates) if remove else 0,
            'remaining': remaining
        }
        
        total_duplicates += len(duplicates)
        
        print(f"      Initial: {initial_count} images")
        print(f"      Duplicates found: {len(duplicates)}")
        if remove:
            print(f"      Removed: {len(duplicates)}")
        print(f"      Remaining: {remaining} images\n")
    
    # Overall summary
    print(f"{'='*60}")
    print(f"✅ Duplicate Detection Complete")
    print(f"{'='*60}")
    print(f"   Total duplicates found: {total_duplicates}")
    if remove:
        print(f"   Total removed: {total_removed}")
    print(f"   Detection method: Perceptual Hash (pHash)")
    print(f"{'='*60}\n")
    
    return stats


def split_training_to_train_val(training_dir, output_dir, val_ratio=0.1, seed=42):
    """
    Split Training directory into Train/Val (90/10)
    
    Reproducing paper methodology:
    - Aiya et al. (2025): "632 images (approximately 10%) were set aside 
      as a validation set"
    
    Args:
        training_dir: Original Training directory
        output_dir: Output directory (train/val will be created)
        val_ratio: Validation ratio (default 0.1 = 10%)
        seed: Random seed for reproducibility
    
    Returns:
        dict: Class-wise split statistics
    """
    print(f"\n{'='*60}")
    print(f"📊 Training → Train/Val Split (90/10)")
    print(f"{'='*60}")
    print(f"   Reference: Aiya et al. (2025) methodology")
    print(f"   Validation ratio: {val_ratio*100:.0f}%\n")
    
    np.random.seed(seed)
    
    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    
    for split_dir in [train_dir, val_dir]:
        os.makedirs(split_dir, exist_ok=True)
    
    stats = {}
    
    # Split each class
    for class_name in sorted(os.listdir(training_dir)):
        class_path = os.path.join(training_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        
        # Create class-specific output directories
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
        
        # List image files
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Shuffle
        np.random.shuffle(image_files)
        
        # Calculate split indices
        n_total = len(image_files)
        n_val = int(n_total * val_ratio)
        n_train = n_total - n_val
        
        train_files = image_files[:n_train]
        val_files = image_files[n_train:]
        
        # Copy files
        for img_file in train_files:
            src = os.path.join(class_path, img_file)
            dst = os.path.join(train_dir, class_name, img_file)
            shutil.copy2(src, dst)
        
        for img_file in val_files:
            src = os.path.join(class_path, img_file)
            dst = os.path.join(val_dir, class_name, img_file)
            shutil.copy2(src, dst)
        
        stats[class_name] = {
            'total': n_total,
            'train': n_train,
            'val': n_val
        }
        
        print(f"   📁 {class_name}:")
        print(f"      Total: {n_total} images")
        print(f"      Train: {n_train} ({n_train/n_total*100:.1f}%)")
        print(f"      Val:   {n_val} ({n_val/n_total*100:.1f}%)\n")
    
    # Overall summary
    total_train = sum(s['train'] for s in stats.values())
    total_val = sum(s['val'] for s in stats.values())
    total_all = total_train + total_val
    
    print(f"{'='*60}")
    print(f"✅ Split Complete")
    print(f"{'='*60}")
    print(f"   Train: {total_train} ({total_train/total_all*100:.1f}%)")
    print(f"   Val:   {total_val} ({total_val/total_all*100:.1f}%)")
    print(f"   Total: {total_all}")
    print(f"{'='*60}\n")
    
    return stats


def prepare_testing_data(testing_dir, output_dir, remove_duplicates=True):
    """
    Process Testing directory: Duplicate removal only (no split)
    
    Args:
        testing_dir: Original Testing directory
        output_dir: Output directory
        remove_duplicates: Whether to remove duplicates
    
    Returns:
        dict: Statistics {class_name: count}
    """
    print(f"\n{'='*60}")
    print(f"🧪 Testing Data Processing")
    print(f"{'='*60}")
    print(f"   Duplicate removal: {'Yes' if remove_duplicates else 'No'}")
    print(f"   Split: None (preserved for final evaluation)\n")
    
    test_output_dir = os.path.join(output_dir, 'test')
    os.makedirs(test_output_dir, exist_ok=True)
    
    # 1. Duplicate detection and removal
    if remove_duplicates:
        dup_stats = find_and_remove_duplicates(
            testing_dir, 
            remove=True, 
            verbose=True
        )
    else:
        dup_stats = {}
    
    # 2. Copy cleaned Testing files
    stats = {}
    
    for class_name in sorted(os.listdir(testing_dir)):
        class_path = os.path.join(testing_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        
        # Create class-specific output directory
        os.makedirs(os.path.join(test_output_dir, class_name), exist_ok=True)
        
        # Copy image files
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_file in image_files:
            src = os.path.join(class_path, img_file)
            dst = os.path.join(test_output_dir, class_name, img_file)
            shutil.copy2(src, dst)
        
        stats[class_name] = len(image_files)
    
    total_test = sum(stats.values())
    
    print(f"{'='*60}")
    print(f"✅ Testing Data Preparation Complete")
    print(f"{'='*60}")
    print(f"   Total images: {total_test}")
    for class_name, count in stats.items():
        print(f"   {class_name}: {count}")
    print(f"{'='*60}\n")
    
    return stats


def preprocess_dataset(raw_data_dir, output_dir, val_ratio=0.1, seed=42):
    """
    Execute full preprocessing pipeline
    
    Workflow:
    1. Training directory:
       - Duplicate removal using perceptual hashing
       - Train/Val split (90/10)
    
    2. Testing directory:
       - Duplicate removal only
       - No split (preserved for final evaluation)
    
    Args:
        raw_data_dir: Raw data directory (contains Training/, Testing/)
        output_dir: Output directory
        val_ratio: Validation ratio (default 0.1 = 10%)
        seed: Random seed for reproducibility
    
    Input Directory Structure:
        raw_data_dir/
        ├── Training/
        │   ├── glioma/
        │   ├── meningioma/
        │   ├── notumor/
        │   └── pituitary/
        └── Testing/
            ├── glioma/
            ├── meningioma/
            ├── notumor/
            └── pituitary/
    
    Output Directory Structure:
        output_dir/
        ├── train/
        │   ├── glioma/
        │   ├── meningioma/
        │   ├── notumor/
        │   └── pituitary/
        ├── val/
        │   ├── glioma/
        │   ├── meningioma/
        │   ├── notumor/
        │   └── pituitary/
        └── test/
            ├── glioma/
            ├── meningioma/
            ├── notumor/
            └── pituitary/
    
    Returns:
        dict: Preprocessing statistics
    """
    print("\n" + "="*60)
    print("🧠 Brain Tumor MRI Dataset Preprocessing")
    print("="*60)
    print(f"   Input directory: {raw_data_dir}")
    print(f"   Output directory: {output_dir}")
    print(f"   Random seed: {seed}")
    print("="*60 + "\n")
    
    training_dir = os.path.join(raw_data_dir, 'Training')
    testing_dir = os.path.join(raw_data_dir, 'Testing')
    
    # Check directory existence
    if not os.path.exists(training_dir):
        raise FileNotFoundError(f"Training directory not found: {training_dir}")
    if not os.path.exists(testing_dir):
        raise FileNotFoundError(f"Testing directory not found: {testing_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # === STEP 1: Process Training directory ===
    print("\n" + "▶"*30)
    print("STEP 1: Training Data Processing")
    print("▶"*30 + "\n")
    
    # 1-1. Duplicate removal
    train_dup_stats = find_and_remove_duplicates(
        training_dir, 
        remove=True, 
        verbose=True
    )
    
    # 1-2. Train/Val split
    train_val_stats = split_training_to_train_val(
        training_dir,
        output_dir,
        val_ratio=val_ratio,
        seed=seed
    )
    
    # === STEP 2: Process Testing directory ===
    print("\n" + "▶"*30)
    print("STEP 2: Testing Data Processing")
    print("▶"*30 + "\n")
    
    test_stats = prepare_testing_data(
        testing_dir,
        output_dir,
        remove_duplicates=True
    )
    
    # === Final Summary ===
    print("\n" + "="*60)
    print("📊 Preprocessing Complete - Final Statistics")
    print("="*60)
    
    # Train statistics
    total_train = sum(s['train'] for s in train_val_stats.values())
    print(f"\n   🟢 Train: {total_train} images")
    for class_name, stats in train_val_stats.items():
        print(f"      {class_name}: {stats['train']}")
    
    # Val statistics
    total_val = sum(s['val'] for s in train_val_stats.values())
    print(f"\n   🟡 Validation: {total_val} images")
    for class_name, stats in train_val_stats.items():
        print(f"      {class_name}: {stats['val']}")
    
    # Test statistics
    total_test = sum(test_stats.values())
    print(f"\n   🔵 Test: {total_test} images")
    for class_name, count in test_stats.items():
        print(f"      {class_name}: {count}")
    
    # Overall
    total_all = total_train + total_val + total_test
    print(f"\n   📦 Total: {total_all} images")
    print(f"      Train:      {total_train} ({total_train/total_all*100:.1f}%)")
    print(f"      Validation: {total_val} ({total_val/total_all*100:.1f}%)")
    print(f"      Test:       {total_test} ({total_test/total_all*100:.1f}%)")
    
    print("\n" + "="*60)
    print("✅ Preprocessing Pipeline Complete!")
    print("="*60 + "\n")
    
    return {
        'train': train_val_stats,
        'test': test_stats,
        'duplicates': {
            'training': train_dup_stats
        }
    }




# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":
    # Configuration
    RAW_DATA_DIR = "data_raw"  # Contains Training/, Testing/ folders
    OUTPUT_DIR = "data"         # Output directory for train/, val/, test/
    
    # Run preprocessing pipeline
    stats = preprocess_dataset(
        raw_data_dir=RAW_DATA_DIR,
        output_dir=OUTPUT_DIR,
        val_ratio=0.1,
        seed=42
    )
    
    print("\n✅ Ready to use:")
    print(f"   Start training: python train.py --data_dir {OUTPUT_DIR}")