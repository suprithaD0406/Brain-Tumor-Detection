import pickle
import matplotlib.pyplot as plt

# ==========================================================
# LOAD TRAINING HISTORY
# ==========================================================

with open(
    "outputs/history.pkl",
    "rb",
) as file:

    history = pickle.load(file)

# ==========================================================
# ACCURACY PLOT
# ==========================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history["accuracy"],
    label="Training Accuracy",
)

plt.plot(
    history["val_accuracy"],
    label="Validation Accuracy",
)

plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()

plt.savefig(
    "outputs/accuracy_plot.png",
    dpi=300,
)

plt.close()

# ==========================================================
# LOSS PLOT
# ==========================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history["loss"],
    label="Training Loss",
)

plt.plot(
    history["val_loss"],
    label="Validation Loss",
)

plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()

plt.savefig(
    "outputs/loss_plot.png",
    dpi=300,
)

plt.close()

print("\nGraphs regenerated successfully!")
print("Saved:")
print("outputs/accuracy_plot.png")
print("outputs/loss_plot.png")