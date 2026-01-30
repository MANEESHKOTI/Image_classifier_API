import os
import shutil
import tarfile
import urllib.request
import random
from config import Config

# Configuration based on PDF Core Requirements
DATASET_URL = "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip?download=1"
RAW_DATA_DIR = "raw_data"
PROCESSED_DATA_DIR = "data"
TRAIN_DIR = os.path.join(PROCESSED_DATA_DIR, "train")
VAL_DIR = os.path.join(PROCESSED_DATA_DIR, "val")
SPLIT_RATIO = 0.8  # 80% train, 20% val as per PDF Step 3

def download_and_extract():
    """
    Downloads the Caltech-101 dataset and extracts it.
    Handles the double compression (zip -> tar.gz) typical of this dataset.
    """
    if not os.path.exists(RAW_DATA_DIR):
        os.makedirs(RAW_DATA_DIR)
    
    # 1. Download the main zip file
    archive_path = os.path.join(RAW_DATA_DIR, "caltech101.zip")
    if not os.path.exists(archive_path):
        print(f"Downloading dataset from {DATASET_URL}...")
        urllib.request.urlretrieve(DATASET_URL, archive_path)
        print("Download complete.")
    
    # 2. Extract the outer zip
    print("Extracting main zip file...")
    shutil.unpack_archive(archive_path, RAW_DATA_DIR)
    
    # 3. Extract the inner tar.gz file
    # Caltech-101 usually extracts to a folder containing '101_ObjectCategories.tar.gz'
    inner_tar_path = os.path.join(RAW_DATA_DIR, "caltech-101", "101_ObjectCategories.tar.gz")
    
    if os.path.exists(inner_tar_path):
        print("Found inner tar.gz. Extracting...")
        with tarfile.open(inner_tar_path, "r:gz") as tar:
            tar.extractall(path=os.path.join(RAW_DATA_DIR, "caltech-101"))
        print("Inner extraction complete.")
    else:
        print("Inner tar.gz not found at expected path. Checking if already extracted...")

def organize_data():
    """
    Organizes data into train/ and val/ directories as required by PDF Core Req 3.
    """
    # Path to the categories after extraction
    source_root = os.path.join(RAW_DATA_DIR, "caltech-101", "101_ObjectCategories")
    
    if not os.path.exists(source_root):
        # Fallback for different extraction behaviors
        potential_paths = [
            os.path.join(RAW_DATA_DIR, "101_ObjectCategories"),
            os.path.join(RAW_DATA_DIR, "caltech-101", "101_ObjectCategories")
        ]
        for p in potential_paths:
            if os.path.exists(p):
                source_root = p
                break
    
    if not os.path.exists(source_root):
        raise FileNotFoundError(f"Could not locate object categories in {RAW_DATA_DIR}")

    # Get all class directories
    classes = [d for d in os.listdir(source_root) if os.path.isdir(os.path.join(source_root, d))]
    
    # Remove clutter if present
    if "BACKGROUND_Google" in classes:
        classes.remove("BACKGROUND_Google")

    # Clean previous runs
    if os.path.exists(PROCESSED_DATA_DIR):
        shutil.rmtree(PROCESSED_DATA_DIR)
    os.makedirs(TRAIN_DIR)
    os.makedirs(VAL_DIR)

    print(f"Processing {len(classes)} classes...")

    for class_name in classes:
        class_dir = os.path.join(source_root, class_name)
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Shuffle for random split
        random.shuffle(images)
        
        split_point = int(len(images) * SPLIT_RATIO)
        train_images = images[:split_point]
        val_images = images[split_point:]
        
        # Create destination folders
        os.makedirs(os.path.join(TRAIN_DIR, class_name), exist_ok=True)
        os.makedirs(os.path.join(VAL_DIR, class_name), exist_ok=True)
        
        # Copy files
        for img in train_images:
            shutil.copy(os.path.join(class_dir, img), os.path.join(TRAIN_DIR, class_name, img))
            
        for img in val_images:
            shutil.copy(os.path.join(class_dir, img), os.path.join(VAL_DIR, class_name, img))
            
    print(f"Data preprocessing complete. Organized into {PROCESSED_DATA_DIR}/train and {PROCESSED_DATA_DIR}/val.")

if __name__ == "__main__":
    download_and_extract()
    organize_data()