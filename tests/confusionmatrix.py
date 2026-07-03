from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

from torch.utils.data import DataLoader
from torch.utils.data import random_split

from model.dataset import TrafficDataset
from model.temporal_cnn import TemporalCNN

from utils.config import (
    DEVICE,
    TRAIN_SPLIT,
    RANDOM_SEED,
    BATCH_SIZE,
    CHECKPOINT_DIR,
    BEST_MODEL_NAME,
)

# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

dataset = TrafficDataset(
    verbose=False,
)

train_size = int(len(dataset) * TRAIN_SPLIT)

validation_size = len(dataset) - train_size

_, validation_dataset = random_split(
    dataset,
    [train_size, validation_size],
    generator=torch.Generator().manual_seed(
        RANDOM_SEED
    ),
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

model = TemporalCNN()

checkpoint = torch.load(
    Path(CHECKPOINT_DIR) / BEST_MODEL_NAME,
    map_location=DEVICE,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(DEVICE)

model.eval()

# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------

true_labels = []

predicted_labels = []

with torch.no_grad():

    for sequences, labels in validation_loader:

        sequences = sequences.to(DEVICE)

        logits = model(sequences)

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        true_labels.extend(
            labels.numpy()
        )

        predicted_labels.extend(
            predictions.cpu().numpy()
        )

# ------------------------------------------------------------
# Report
# ------------------------------------------------------------

class_names = [
    "Chat",
    "Email",
    "FileTransfer",
    "P2P",
    "Streaming",
    "VoIP",
]

print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(

    classification_report(
        true_labels,
        predicted_labels,
        target_names=class_names,
        digits=4,
    )
)

# ------------------------------------------------------------
# Confusion Matrix
# ------------------------------------------------------------

cm = confusion_matrix(
    true_labels,
    predicted_labels,
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names,
)

fig, ax = plt.subplots(
    figsize=(8,8)
)

disp.plot(
    ax=ax,
    cmap="Blues",
    colorbar=False,
)

plt.title(
    "Temporal CNN Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300,
)

plt.show()