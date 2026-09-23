from pathlib import Path
from PIL import Image

# Path to the dataset
dataset_path = Path("dataset")

# Check whether the dataset exists
if not dataset_path.exists():
    print("❌ Dataset folder not found!")
    exit()

print("✅ Dataset found successfully!\n")

total_images = 0

print("Dataset Summary")
print("-" * 30)

# Loop through each class folder
for folder in dataset_path.iterdir():

    if folder.is_dir():

        image_files = list(folder.glob("*"))

        image_count = len(image_files)

        total_images += image_count

        print(f"{folder.name:<15} : {image_count} images")

print("-" * 30)
print(f"Total Images   : {total_images}")