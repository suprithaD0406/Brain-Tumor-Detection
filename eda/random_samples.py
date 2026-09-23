from pathlib import Path
import random
import matplotlib.pyplot as plt
from PIL import Image

DATASET_PATH = Path("dataset")

random.seed(42)

classes = sorted([folder for folder in DATASET_PATH.iterdir() if folder.is_dir()])

fig, axes = plt.subplots(
    len(classes),
    5,
    figsize=(15, 12)
)

for row, class_folder in enumerate(classes):

    images = [
        img
        for img in class_folder.iterdir()
        if img.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]

    samples = random.sample(images, 5)

    for col, image_path in enumerate(samples):

        image = Image.open(image_path)

        axes[row][col].imshow(image)

        axes[row][col].axis("off")

        if col == 0:
            axes[row][col].set_ylabel(
                class_folder.name,
                fontsize=12,
                rotation=90,
            )

plt.tight_layout()

plt.show()
