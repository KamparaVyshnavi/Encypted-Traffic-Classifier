# Dataset

import torch

DEFAULT_SEQUENCE_LENGTH = 20
FEATURE_DIMENSION = 6
NUM_CLASSES = 6

# CNN Architecture

CNN_CHANNELS = [64, 128, 256]
KERNEL_SIZE = 3
STRIDE = 1
PADDING = 1

CONV_DROPOUT = 0.2
FC_HIDDEN = 128
FC_DROPOUT = 0.3

# Tensor

TENSOR_DTYPE = torch.float32

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PIN_MEMORY = True
NON_BLOCKING = True

# ----------------------------------------------------------
# Training
# ----------------------------------------------------------

BATCH_SIZE = 64

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

NUM_EPOCHS = 100

TRAIN_SPLIT = 0.8

VALIDATION_SPLIT = 0.2

RANDOM_SEED = 42

CHECKPOINT_DIR = "model/saved_models/normalised_final"

BEST_MODEL_NAME = "best_model.pth"
LATEST_MODEL_NAME = "latest_model.pth"

DATASET_ROOT = "datasets/processed_datasets/iscx_normalised"

# ----------------------------------------------------------
# Cross Network Evaluation
# ----------------------------------------------------------

# Baseline Experiment

BASELINE_MODEL_PATH = (
    "model/saved_models/baseline_final/best_model.pth"
)

BASELINE_TEST_DATASET = (
    "datasets/processed_datasets/vnat_raw"
)

# Novelty-1 Experiment

NOVELTY_MODEL_PATH = (
    "model/saved_models/normalised_final/best_model.pth"
)

NOVELTY_TEST_DATASET = (
    "datasets/processed_datasets/vnat_normalised"
)


LABEL_MAPPING = {
    "Chat": 0,
    "Email": 1,
    "FileTransfer": 2,
    "P2P": 3,
    "Streaming": 4,
    "VoIP": 5,
}

INDEX_TO_LABEL = {
    value: key
    for key, value in LABEL_MAPPING.items()
}


# ---------------------------------------------------------
# Multi-Exit Checkpoints
# ---------------------------------------------------------

MULTI_EXIT_CHECKPOINT_DIR = "model/saved_models/multi_exit_normalised"

MULTI_EXIT_BEST_MODEL_NAME = "best_model.pth"

MULTI_EXIT_LATEST_MODEL_NAME = "latest_model.pth"

MULTI_EXIT_HISTORY_NAME = "training_history.json"

# ---------------------------------------------------------
# Multi-Exit Inference
# ---------------------------------------------------------

ISCX_THRESHOLD = 0.90

VNAT_THRESHOLD = 0.70