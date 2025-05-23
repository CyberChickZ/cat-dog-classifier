import os
import shutil
import random
from sklearn.model_selection import train_test_split
import kagglehub
from PIL import Image, UnidentifiedImageError

# Parameters
KAGGLE_DATASET = "bhavikjikadara/dog-and-cat-classification-dataset"
CLASSES = ["Cat", "Dog"]
SPLIT_RATIOS = [0.7, 0.2, 0.1]  # train, val, test

# Directory definitions
RAW_ROOT_DIR = "data_name"       # Original download directory (e.g. data_name/Cat/*.jpg)
TARGET_ROOT_DIR = "data"         # Final organized directory (e.g. data/train/Cat)

def download_if_needed():
    if not os.path.exists(RAW_ROOT_DIR):
        print("Data does not exist, downloading from Kaggle...")
        path = kagglehub.dataset_download(KAGGLE_DATASET)
        print("Download complete, path is:", path)
    else:
        print("Data already exists, skipping download")

def make_directories():
    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            dir_path = os.path.join(TARGET_ROOT_DIR, split, cls)
            os.makedirs(dir_path, exist_ok=True)

def is_valid_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except (IOError, UnidentifiedImageError):
        return False

def split_and_copy(class_name):
    src = os.path.join(RAW_ROOT_DIR, class_name)
    images = [f for f in os.listdir(src) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    images.sort()

    # Split data
    train, temp = train_test_split(images, test_size=1 - SPLIT_RATIOS[0], random_state=42)
    val, test = train_test_split(temp, test_size=SPLIT_RATIOS[2] / (SPLIT_RATIOS[1] + SPLIT_RATIOS[2]), random_state=42)

    def copy_files(file_list, split_name):
        dest = os.path.join(TARGET_ROOT_DIR, split_name, class_name)
        for fname in file_list:
            src_path = os.path.join(src, fname)
            if not is_valid_image(src_path):
                print(f"Skipped corrupted image: {src_path}")
                continue
            shutil.copy(src_path, os.path.join(dest, fname))

    copy_files(train, "train")
    copy_files(val, "val")
    copy_files(test, "test")

if __name__ == "__main__":
    download_if_needed()
    make_directories()
    for cls in CLASSES:
        split_and_copy(cls)
    print("Data set split completed! Structure is as follows:\ndata/train/[Cat|Dog] ...")