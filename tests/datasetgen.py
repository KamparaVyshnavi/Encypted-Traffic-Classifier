from pathlib import Path

import numpy as np


class DatasetValidator:

    def __init__(self, dataset_dir):

        self.dataset_dir = Path(dataset_dir)

        self.sequence_dir = (
            self.dataset_dir /
            "sequences"
        )

        self.labels_file = (
            self.dataset_dir /
            "labels.csv"
        )

        self.metadata_file = (
            self.dataset_dir /
            "metadata.json"
        )

        self.total_samples = 0

        self.expected_shape = None

        self.failed = 0

    # =====================================================
    # Main
    # =====================================================

    def run(self):

        print()

        print("=" * 60)
        print("DATASET VALIDATION")
        print("=" * 60)

        print()

        self.check_directories()

        self.check_sequences()

        self.check_labels()

        self.check_metadata()

        self.print_summary()

    # =====================================================
    # Directory Checks
    # =====================================================

    def check_directories(self):

        if not self.dataset_dir.exists():

            raise FileNotFoundError(
                self.dataset_dir
            )

        if not self.sequence_dir.exists():

            raise FileNotFoundError(
                self.sequence_dir
            )

        if not self.labels_file.exists():

            raise FileNotFoundError(
                self.labels_file
            )

        if not self.metadata_file.exists():

            raise FileNotFoundError(
                self.metadata_file
            )

    # =====================================================
    # Sequence Checks
    # =====================================================

    def check_sequences(self):

        files = sorted(

            self.sequence_dir.glob(
                "*.npy"
            )

        )

        self.total_samples = len(files)

        print(
            f"Sequence Files : {self.total_samples}"
        )

        if self.total_samples == 0:

            raise RuntimeError(
                "No dataset generated."
            )

        for file in files:

            self.validate_sample(file)

    # =====================================================
    # Sample Validation
    # =====================================================

    def validate_sample(
        self,
        file
    ):

        try:

            sample = np.load(file)

        except Exception:

            print(
                f"Cannot read {file.name}"
            )

            self.failed += 1

            return

        if sample.ndim != 2:

            print(
                f"{file.name} not 2D"
            )

            self.failed += 1

            return

        if self.expected_shape is None:

            self.expected_shape = sample.shape

        if sample.shape != self.expected_shape:

            print(
                f"{file.name} shape mismatch "
                f"{sample.shape}"
            )

            self.failed += 1

        if np.isnan(sample).any():

            print(
                f"{file.name} contains NaN"
            )

            self.failed += 1

        if np.isinf(sample).any():

            print(
                f"{file.name} contains Inf"
            )

            self.failed += 1

    # =====================================================
    # Labels
    # =====================================================

    def check_labels(self):

        with open(
            self.labels_file,
            "r"
        ) as fp:

            rows = fp.readlines()

        total_labels = len(rows) - 1

        print(
            f"Labels : {total_labels}"
        )

        if total_labels != self.total_samples:

            raise RuntimeError(

                "Labels and samples "
                "do not match."

            )

    # =====================================================
    # Metadata
    # =====================================================

    def check_metadata(self):

        import json

        with open(
            self.metadata_file,
            "r"
        ) as fp:

            metadata = json.load(fp)

        print()

        print(
            "Metadata"
        )

        print(
            "--------"
        )

        print(

            "Feature Dimension :",

            metadata[
                "feature_dimension"
            ]

        )

        print(

            "Sequence Length   :",

            metadata[
                "sequence_length"
            ]

        )

        print(

            "Classes           :",

            len(
                metadata[
                    "classes"
                ]
            )

        )

        print(

            "Samples Saved     :",

            metadata[
                "samples_saved"
            ]

        )

    # =====================================================
    # Summary
    # =====================================================

    def print_summary(self):

        print()

        print("=" * 60)

        print("VALIDATION SUMMARY")

        print("=" * 60)

        print()

        print(

            "Expected Shape :",

            self.expected_shape

        )

        print(

            "Samples Checked:",

            self.total_samples

        )

        print(

            "Failures :",

            self.failed

        )

        print()

        if self.failed == 0:

            print(
                "Dataset validation PASSED"
            )

        else:

            print(
                "Dataset validation FAILED"
            )


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    validator = DatasetValidator(

        "datasets/processed_sequences"

    )

    validator.run()