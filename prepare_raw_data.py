import os
import shutil
import numpy as np

def split_without_dedup(raw_dir, output_dir, val_ratio=0.1, seed=42):
    np.random.seed(seed)
    training_dir = os.path.join(raw_dir, 'Training')
    testing_dir = os.path.join(raw_dir, 'Testing')

    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')

    for d in [train_dir, val_dir, test_dir]:
        os.makedirs(d, exist_ok=True)

    for class_name in sorted(os.listdir(training_dir)):
        class_path = os.path.join(training_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

        files = sorted([f for f in os.listdir(class_path)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        np.random.shuffle(files)

        n_val = int(len(files) * val_ratio)
        for f in files[n_val:]:
            shutil.copy2(os.path.join(class_path, f),
                        os.path.join(train_dir, class_name, f))
        for f in files[:n_val]:
            shutil.copy2(os.path.join(class_path, f),
                        os.path.join(val_dir, class_name, f))

    for class_name in sorted(os.listdir(testing_dir)):
        class_path = os.path.join(testing_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)
        for f in os.listdir(class_path):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                shutil.copy2(os.path.join(class_path, f),
                            os.path.join(test_dir, class_name, f))

    print("Done. Stats:")
    for split in ['train', 'val', 'test']:
        total = sum(len(os.listdir(os.path.join(output_dir, split, c)))
                   for c in os.listdir(os.path.join(output_dir, split)))
        print(f"  {split}: {total}")

if __name__ == "__main__":
    split_without_dedup("data_raw", "data_raw_split")
