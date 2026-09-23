from pathlib import Path
from PIL import Image
from collections import Counter

DATASET_PATH = Path("dataset")

image_sizes = []
image_modes = []
corrupted_images = []

total_images = 0

for class_folder in DATASET_PATH.iterdir():

    if not class_folder.is_dir():
        continue

    for image_path in class_folder.iterdir():

        if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        total_images += 1

        try:
            with Image.open(image_path) as img:

                image_sizes.append(img.size)
                image_modes.append(img.mode)

        except Exception:

            corrupted_images.append(str(image_path))

# =========================================

size_counter = Counter(image_sizes)
mode_counter = Counter(image_modes)

print("\n========== IMAGE PROPERTY REPORT ==========\n")

print(f"Total Images : {total_images}\n")

print("Image Modes")
print("----------------------------------")

for mode, count in mode_counter.items():
    print(f"{mode:<10} : {count}")

print("\nMost Common Image Sizes")
print("----------------------------------")

for size, count in size_counter.most_common(10):
    print(f"{size} : {count}")

print("\nSmallest Image")
print("----------------------------------")
print(min(image_sizes))

print("\nLargest Image")
print("----------------------------------")
print(max(image_sizes))

print("\nCorrupted Images")
print("----------------------------------")
print(len(corrupted_images))

if corrupted_images:

    print("\nList of Corrupted Images\n")

    for img in corrupted_images:
        print(img)