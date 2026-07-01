"""
=========================================================================
Test : Training Pipeline
=========================================================================
"""

import sys
from pathlib import Path

# ----------------------------------------------------------------------
# Add Project Root
# ----------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ----------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------

from model.train import (
    set_random_seed,
    create_dataloaders,
    build_model,
    create_optimizer,
    create_loss,
    train_one_epoch,
    validate,
    print_epoch_summary,
)

from utils.config import RANDOM_SEED

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main():

    print("=" * 70)
    print("TRAINING PIPELINE TEST")
    print("=" * 70)

    # ------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------

    print("\n[1] Initializing...")

    set_random_seed(RANDOM_SEED)

    train_loader, validation_loader = create_dataloaders()

    model = build_model()

    optimizer = create_optimizer(model)

    criterion = create_loss()

    print("✓ Initialization complete.")

    print(f"Training Samples   : {len(train_loader.dataset)}")
    print(f"Validation Samples : {len(validation_loader.dataset)}")

    # ------------------------------------------------------------
    # One Epoch
    # ------------------------------------------------------------

    print("\n[2] Running One Training Epoch...")

    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader,
        optimizer,
        criterion,
    )

    validation_loss, validation_accuracy = validate(
        model,
        validation_loader,
        criterion,
    )

    print_epoch_summary(
        1,
        train_loss,
        train_accuracy,
        validation_loss,
        validation_accuracy,
    )

    # ------------------------------------------------------------
    # Success
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING PIPELINE TEST PASSED")
    print("=" * 70)


# ----------------------------------------------------------------------

if __name__ == "__main__":
    main()