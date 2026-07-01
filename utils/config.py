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

NUM_EPOCHS = 30

TRAIN_SPLIT = 0.8

VALIDATION_SPLIT = 0.2

RANDOM_SEED = 42

CHECKPOINT_DIR = "model/saved_models/normalised"

BEST_MODEL_NAME = "best_model.pth"
LATEST_MODEL_NAME = "latest_model.pth"

DATASET_ROOT = "datasets/processed_sequences_norm"