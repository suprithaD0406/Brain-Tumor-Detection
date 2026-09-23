from pathlib import Path
from PIL import Image
import numpy as np

DATASET_PATH = Path("dataset")

print("\n========== BRIGHTNESS ANALYSIS ==========\n")

for class_folder in sorted(DATASET_PATH.iterdir()):

    if not class_folder.is_dir():
        continue

    brightness = []

    for image_path in class_folder.iterdir():

        if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        image = Image.open(image_path).convert("L")

        arr = np.array(image)

        brightness.append(arr.mean())

    print(f"{class_folder.name:<15} Average Brightness : {np.mean(brightness):.2f}")
    