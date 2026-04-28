import os
import shutil
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import imagehash


def get_perceptual_hash(image_path, hash_size=8):
    img = Image.open(image_path)
    img_hash = imagehash.phash(img, hash_size=hash_size)
    return str(img_hash)


def find_and_remove_duplicates(data_dir, remove=True, verbose=True):
    print(f"\n{'='*60}")
    print(f"Duplicate Image Detection: {os.path.basename(data_dir)}")
    print(f"{'='*60}\n")

    stats = {}
    total_duplicates = 0
    total_removed = 0

    for class_name in sorted(os.listdir(data_dir)):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        print(f"  {class_name}")

        image_files = [f for f in os.listdir(class_path)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        initial_count = len(image_files)
        hash_dict = {}
        duplicates = []

        for img_file in image_files:
            img_path = os.path.join(class_path, img_file)
            try:
                img_hash = get_perceptual_hash(img_path)
                if img_hash in hash_dict:
                    duplicates.append((img_file, hash_dict[img_hash]))
                    if verbose:
                        print(f"    Duplicate: {img_file} ~ {hash_dict[img_hash]}")
                    if remove:
                        os.remove(img_path)
                        total_removed += 1
                else:
                    hash_dict[img_hash] = img_file
            except Exception as e:
                print(f"    Error processing {img_file}: {e}")

        remaining = len([f for f in os.listdir(class_path)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

        stats[class_name] = {
            'initial': initial_count,
            'duplicates_found': len(duplicates),
            'duplicates_removed': len(duplicates) if remove else 0,
            'remaining': remaining
        }

        total_duplicates += len(duplicates)

        print(f"    Initial:    {initial_count}")
        print(f"    Duplicates: {len(duplicates)}")
        if remove:
            print(f"    Removed:    {len(duplicates)}")
        print(f"    Remaining:  {remaining}\n")

    print(f"{'='*60}")
    print(f"Total duplicates found: {total_duplicates}")
    if remove:
        print(f"Total removed: {total_removed}")
    print(f"{'='*60}\n")

    return stats


def split_training_to_train_val(training_dir, output_dir, val_ratio=0.1, seed=42):
    print(f"\n{'='*60}")
    print(f"Train/Val Split ({int((1-val_ratio)*100)}/{int(val_ratio*100)})")
    print(f"{'='*60}\n")

    np.random.seed(seed)

    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')

    for split_dir in [train_dir, val_dir]:
        os.makedirs(split_dir, exist_ok=True)

    stats = {}

    for class_name in sorted(os.listdir(training_dir)):
        class_path = os.path.join(training_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

        image_files = [f for f in os.listdir(class_path)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        np.random.shuffle(image_files)

        n_total = len(image_files)
        n_val = int(n_total * val_ratio)
        n_train = n_total - n_val

        train_files = image_files[:n_train]
        val_files = image_files[n_train:]

        for img_file in train_files:
            shutil.copy2(os.path.join(class_path, img_file),
                         os.path.join(train_dir, class_name, img_file))

        for img_file in val_files:
            shutil.copy2(os.path.join(class_path, img_file),
                         os.path.join(val_dir, class_name, img_file))

        stats[class_name] = {
            'total': n_total,
            'train': n_train,
            'val': n_val
        }

        print(f"  {class_name}: total={n_total}, train={n_train}, val={n_val}")

    total_train = sum(s['train'] for s in stats.values())
    total_val = sum(s['val'] for s in stats.values())
    total_all = total_train + total_val

    print(f"\n{'='*60}")
    print(f"Train: {total_train} ({total_train/total_all*100:.1f}%)")
    print(f"Val:   {total_val} ({total_val/total_all*100:.1f}%)")
    print(f"Total: {total_all}")
    print(f"{'='*60}\n")

    return stats


def prepare_testing_data(testing_dir, output_dir, remove_duplicates=True):
    print(f"\n{'='*60}")
    print(f"Testing Data Processing")
    print(f"{'='*60}\n")

    test_output_dir = os.path.join(output_dir, 'test')
    os.makedirs(test_output_dir, exist_ok=True)

    if remove_duplicates:
        dup_stats = find_and_remove_duplicates(testing_dir, remove=True, verbose=True)
    else:
        dup_stats = {}

    stats = {}

    for class_name in sorted(os.listdir(testing_dir)):
        class_path = os.path.join(testing_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        os.makedirs(os.path.join(test_output_dir, class_name), exist_ok=True)

        image_files = [f for f in os.listdir(class_path)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        for img_file in image_files:
            shutil.copy2(os.path.join(class_path, img_file),
                         os.path.join(test_output_dir, class_name, img_file))

        stats[class_name] = len(image_files)

    total_test = sum(stats.values())

    print(f"{'='*60}")
    print(f"Total test images: {total_test}")
    for class_name, count in stats.items():
        print(f"  {class_name}: {count}")
    print(f"{'='*60}\n")

    return stats


def preprocess_dataset(raw_data_dir, output_dir, val_ratio=0.1, seed=42):
    print("\n" + "="*60)
    print("Brain Tumor MRI Dataset Preprocessing")
    print("="*60)
    print(f"  Input:  {raw_data_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Seed:   {seed}")
    print("="*60 + "\n")

    training_dir = os.path.join(raw_data_dir, 'Training')
    testing_dir = os.path.join(raw_data_dir, 'Testing')

    if not os.path.exists(training_dir):
        raise FileNotFoundError(f"Training directory not found: {training_dir}")
    if not os.path.exists(testing_dir):
        raise FileNotFoundError(f"Testing directory not found: {testing_dir}")

    os.makedirs(output_dir, exist_ok=True)

    print("STEP 1: Training Data")
    print("-"*30)
    train_dup_stats = find_and_remove_duplicates(training_dir, remove=True, verbose=True)
    train_val_stats = split_training_to_train_val(training_dir, output_dir,
                                                   val_ratio=val_ratio, seed=seed)

    print("STEP 2: Testing Data")
    print("-"*30)
    test_stats = prepare_testing_data(testing_dir, output_dir, remove_duplicates=True)

    print("\n" + "="*60)
    print("Final Statistics")
    print("="*60)

    total_train = sum(s['train'] for s in train_val_stats.values())
    total_val = sum(s['val'] for s in train_val_stats.values())
    total_test = sum(test_stats.values())
    total_all = total_train + total_val + total_test

    print(f"\n  Train:      {total_train} ({total_train/total_all*100:.1f}%)")
    for class_name, s in train_val_stats.items():
        print(f"    {class_name}: {s['train']}")

    print(f"\n  Validation: {total_val} ({total_val/total_all*100:.1f}%)")
    for class_name, s in train_val_stats.items():
        print(f"    {class_name}: {s['val']}")

    print(f"\n  Test:       {total_test} ({total_test/total_all*100:.1f}%)")
    for class_name, count in test_stats.items():
        print(f"    {class_name}: {count}")

    print(f"\n  Total:      {total_all}")
    print("="*60 + "\n")

    return {
        'train': train_val_stats,
        'test': test_stats,
        'duplicates': {'training': train_dup_stats}
    }


if __name__ == "__main__":
    RAW_DATA_DIR = "data_raw"
    OUTPUT_DIR = "data"

    stats = preprocess_dataset(
        raw_data_dir=RAW_DATA_DIR,
        output_dir=OUTPUT_DIR,
        val_ratio=0.1,
        seed=42
    )

    print(f"Done. Start training: python train.py --data_dir {OUTPUT_DIR}")