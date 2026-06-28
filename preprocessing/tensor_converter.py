from dataclasses import dataclass
from typing import Any, List, Optional

import torch

from flow.flow_table import FlowKey
from preprocessing.feature_encoder import EncodedSequence


@dataclass
class TensorSequence:
    """
    Output produced by TensorConverter.

    Tensor Shape

        sequence_length × feature_dimension
    """

    flow_key: FlowKey

    tensor: torch.Tensor

    feature_dimension: int
    sequence_length: int

    valid_exit_points: List[int]

    is_padded: bool
    is_closed: bool


@dataclass
class TensorConverterConfig:
    """
    Tensor conversion configuration.
    """

    dtype: torch.dtype = torch.float32
    device: str = "cpu"


class TensorConverter:
    """
    Converts encoded feature matrices into
    PyTorch tensors.

    Compatible with

    - FeatureEncoder
    - future Dataset
    - future TemporalCNN
    - future MultiExitCNN
    - future Training pipeline
    - future Inference pipeline
    """

    def __init__(
        self,
        config: Optional[TensorConverterConfig] = None,
        dataset: Optional[Any] = None,
    ) -> None:

        self.config = (
            config
            or TensorConverterConfig()
        )

        self.dataset = dataset

    # --------------------------------------------------
    # Main API
    # --------------------------------------------------

    def convert_sequence(
        self,
        encoded_sequence: EncodedSequence,
    ) -> TensorSequence:

        self._validate(encoded_sequence)

        tensor = torch.tensor(
            encoded_sequence.features,
            dtype=self.config.dtype,
            device=self.config.device,
        )

        return TensorSequence(

            flow_key=
                encoded_sequence.flow_key,

            tensor=
                tensor,

            feature_dimension=
                encoded_sequence.feature_dimension,

            sequence_length=
                encoded_sequence.sequence_length,

            valid_exit_points=
                list(
                    encoded_sequence.valid_exit_points
                ),

            is_padded=
                encoded_sequence.is_padded,

            is_closed=
                encoded_sequence.is_closed,
        )

    def convert_sequences(
        self,
        encoded_sequences:
        List[EncodedSequence],
    ) -> List[TensorSequence]:

        return [

            self.convert_sequence(
                encoded_sequence
            )

            for encoded_sequence
            in encoded_sequences
        ]

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    def _validate(
        self,
        encoded_sequence:
        EncodedSequence,
    ) -> None:

        if encoded_sequence is None:
            raise ValueError(
                "EncodedSequence cannot be None."
            )

        if (
            encoded_sequence.features
            is None
        ):
            raise ValueError(
                "Features cannot be None."
            )

        if len(
            encoded_sequence.features
        ) != encoded_sequence.sequence_length:

            raise ValueError(

                "Sequence length mismatch."
            )

        for feature_vector in (
            encoded_sequence.features
        ):

            if len(feature_vector) != (
                encoded_sequence.feature_dimension
            ):

                raise ValueError(

                    "Feature dimension mismatch."
                )

    # --------------------------------------------------
    # Forward Integration Hooks
    # --------------------------------------------------

    def try_dataset(
        self,
        tensor_sequence:
        TensorSequence,
    ) -> Any:

        if self.dataset is None:
            return tensor_sequence

        if hasattr(
            self.dataset,
            "receive_tensor_sequence",
        ):

            return self.dataset.\
                receive_tensor_sequence(
                    tensor_sequence
                )

        return tensor_sequence

    # --------------------------------------------------
    # Utility Methods
    # --------------------------------------------------

    def get_dtype(
        self,
    ):

        return self.config.dtype

    def get_device(
        self,
    ):

        return self.config.device