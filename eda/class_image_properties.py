from pathlib import Path
from collections import Counter
from PIL import Image

DATASET_PATH = Path("dataset")

print("\n========== CLASS-WISE IMAGE REPORT ==========\n")

for class_folder in sorted(DATASET_PATH.iterdir()):

    if not class_folder.is_dir():
        continue

    size_counter = Counter()
    mode_counter = Counter()
    total = 0

    for image_path in class_folder.iterdir():

        if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        try:

            with Image.open(image_path) as img:

                total += 1

                size_counter[img.size] += 1
                mode_counter[img.mode] += 1

        except:
            pass

    print("=" * 50)
    print(f"Class : {class_folder.name}")
    print(f"Total Images : {total}")

    print("\nImage Modes")

    for mode, count in mode_counter.items():
        print(f"   {mode:<5} : {count}")

    print("\nTop 5 Image Sizes")

    for size, count in size_counter.most_common(5):
        print(f"   {size} : {count}")

    print()