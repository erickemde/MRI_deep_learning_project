import os
import shutil
from PIL import Image
import imagehash


def get_perceptual_hash(image_path, hash_size=8):
    img = Image.open(image_path)
    img_hash = imagehash.phash(img, hash_size=hash_size)
    return str(img_hash)


def find_and_remove_duplicates(data_dir, remove=True, verbose=True):
    print(f"\n{'='*60}")
    print(f"Duplicate Detection: {os.path.basename(data_dir)}")
    print(f"{'='*60}\n")

    total_duplicates = 0
    stats = {}

    for class_name in sorted(os.listdir(data_dir)):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        print(f"  {class_name}")
        image_files = sorted([f for f in os.listdir(class_path)
                               if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

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
                else:
                    hash_dict[img_hash] = img_file
            except Exception as e:
                print(f"    Error processing {img_file}: {e}")

        remaining = len([f for f in os.listdir(class_path)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

        stats[class_name] = {
            'initial': initial_count,
            'duplicates': len(duplicates),
            'remaining': remaining
        }

        total_duplicates += len(duplicates)
        print(f"    Initial:    {initial_count}")
        print(f"    Duplicates: {len(duplicates)}")
        print(f"    Remaining:  {remaining}\n")

    print(f"{'='*60}")
    print(f"Total duplicates found: {total_duplicates}")
    print(f"{'='*60}\n")
    return stats


def copy_split(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    for class_name in sorted(os.listdir(src_dir)):
        class_path = os.path.join(src_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        dst_class = os.path.join(dst_dir, class_name)
        os.makedirs(dst_class, exist_ok=True)
        for img_file in os.listdir(class_path):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                shutil.copy2(os.path.join(class_path, img_file),
                             os.path.join(dst_class, img_file))


def preprocess_xray_dataset(raw_data_dir, output_dir):
    print("\n" + "="*60)
    print("Chest X-Ray Dataset Preprocessing")
    print("="*60)
    print(f"  Input:  {raw_data_dir}")
    print(f"  Output: {output_dir}")
    print("="*60 + "\n")

    # X-Ray 데이터셋 경로
    xray_dir = os.path.join(raw_data_dir, "chest_xray")
    train_src = os.path.join(xray_dir, "train")
    val_src   = os.path.join(xray_dir, "val")
    test_src  = os.path.join(xray_dir, "test")

    os.makedirs(output_dir, exist_ok=True)

    # 중복 제거
    print("STEP 1: Duplicate Removal")
    print("-"*30)
    for split_name, split_path in [("train", train_src),
                                    ("val",   val_src),
                                    ("test",  test_src)]:
        print(f"\n[{split_name}]")
        find_and_remove_duplicates(split_path, remove=True, verbose=True)

    # 데이터 복사
    print("STEP 2: Copying to output directory")
    print("-"*30)
    for split_name, split_path in [("train", train_src),
                                    ("val",   val_src),
                                    ("test",  test_src)]:
        dst = os.path.join(output_dir, split_name)
        copy_split(split_path, dst)
        print(f"  {split_name} copied to {dst}")

    # 최종 통계
    print("\n" + "="*60)
    print("Final Statistics")
    print("="*60)
    total_all = 0
    for split_name in ["train", "val", "test"]:
        split_path = os.path.join(output_dir, split_name)
        split_total = 0
        for class_name in sorted(os.listdir(split_path)):
            class_path = os.path.join(split_path, class_name)
            count = len([f for f in os.listdir(class_path)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            print(f"  {split_name}/{class_name}: {count}")
            split_total += count
        print(f"  {split_name} total: {split_total}\n")
        total_all += split_total
    print(f"  Grand Total: {total_all}")
    print("="*60 + "\n")


if __name__ == "__main__":
    RAW_DATA_DIR = "data_raw_xray"
    OUTPUT_DIR   = "data_xray"

    preprocess_xray_dataset(
        raw_data_dir=RAW_DATA_DIR,
        output_dir=OUTPUT_DIR
    )

    print("Done. Start training: python train.py --data_dir data_xray")
