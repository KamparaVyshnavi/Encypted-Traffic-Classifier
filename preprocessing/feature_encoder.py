from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from flow.flow_table import FlowKey
from flow.sequence_builder import SequenceRecord


@dataclass
class EncodedSequence:
    """
    Output produced by FeatureEncoder.

    Features shape:

        sequence_length × feature_dimension

    Current feature order:

    [
    packet_len,
    direction,
    protocol_id,
    tcp_flags,
    relative_timestamp,
    inter_arrival_time
    ]
    """

    flow_key: FlowKey
    features: List[List[float]]

    feature_dimension: int
    sequence_length: int

    valid_exit_points: List[int]

    is_padded: bool
    is_closed: bool


@dataclass
class FeatureEncoderConfig:
    """
    Feature encoding configuration.
    """

    protocol_mapping: Optional[Dict[str, int]] = None
    use_normalized_features: bool = False

    def __post_init__(self) -> None:
        if self.protocol_mapping is None:
            self.protocol_mapping = {
                "PAD": 0,
                "TCP": 1,
                "UDP": 2,
            }


class FeatureEncoder:
    """
    Converts SequenceRecord objects into numerical
    feature matrices suitable for ML models.

    Compatible with:
    - SequenceBuilder
    - future HandshakeDetector
    - future Normalizer
    - future TensorConverter
    - future Dataset
    - future Training pipeline
    - future Inference pipeline
    """

    FEATURE_DIMENSION = 6

    def __init__(
        self,
        config: Optional[FeatureEncoderConfig] = None,
        normalizer: Optional[Any] = None,
        tensor_converter: Optional[Any] = None,
    ) -> None:

        self.config = config or FeatureEncoderConfig()

        self.normalizer = normalizer
        self.tensor_converter = tensor_converter

    # --------------------------------------------------
    # Main Encoding API
    # --------------------------------------------------

    def encode_sequence(
        self,
        sequence_record: SequenceRecord,
    ) -> EncodedSequence:

        if self.config.use_normalized_features:

            packets = getattr(
                sequence_record,
                "normalized_sequence",
                None,
            )

            if packets is None:
                packets = getattr(
                    sequence_record,
                    "raw_sequence",
                    sequence_record.sequence,
                )

        else:

            packets = getattr(
                sequence_record,
                "raw_sequence",
                None,
            )

            if packets is None:
                packets = sequence_record.sequence

        features = []

        for packet in packets:
            features.append(
                self.encode_packet(packet)
            )

        return EncodedSequence(
            flow_key=sequence_record.flow_key,
            features=features,
            feature_dimension=self.FEATURE_DIMENSION,
            sequence_length=sequence_record.sequence_length,
            valid_exit_points=(
                sequence_record.valid_exit_points
            ),
            is_padded=sequence_record.is_padded,
            is_closed=sequence_record.is_closed,
        )

    def encode_sequences(
        self,
        sequence_records: List[SequenceRecord],
    ) -> List[EncodedSequence]:

        return [
            self.encode_sequence(sequence_record)
            for sequence_record in sequence_records
        ]

    # --------------------------------------------------
    # Packet Encoding
    # --------------------------------------------------

    def encode_packet(
    self,
    packet: Dict[str, Any],
) -> List[float]:

        packet_len = float(
            packet.get(
                "packet_len",
                0
            )
        )

        direction = float(
            packet.get(
                "direction",
                0
            ) or 0
        )

        protocol_id = float(

            self._protocol_to_id(

                packet.get(
                    "protocol"
                )
            )
        )

        tcp_flags = float(

            packet.get(
                "tcp_flags",
                0
            ) or 0
        )

        relative_timestamp = float(

            packet.get(
                "relative_timestamp",
                0.0
            )
        )

        inter_arrival_time = float(

            packet.get(
                "inter_arrival_time",
                0.0
            )
        )

        return [

            packet_len,

            direction,

            protocol_id,

            tcp_flags,

            relative_timestamp,

            inter_arrival_time,
        ]
    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------

    def _protocol_to_id(
        self,
        protocol: Optional[str],
    ) -> int:

        if protocol is None:
            return 0

        protocol = protocol.upper()

        return self.config.protocol_mapping.get(
            protocol,
            0,
        )

    # --------------------------------------------------
    # Future Integration Hooks
    # --------------------------------------------------

    def try_normalize(
        self,
        encoded_sequence: EncodedSequence,
    ) -> Any:

        if self.normalizer is None:
            return encoded_sequence

        if hasattr(
            self.normalizer,
            "normalize_sequence",
        ):
            return self.normalizer.normalize_sequence(
                encoded_sequence
            )

        return encoded_sequence

    def try_convert_tensor(
        self,
        encoded_sequence: EncodedSequence,
    ) -> Any:

        if self.tensor_converter is None:
            return encoded_sequence

        if hasattr(
            self.tensor_converter,
            "convert_sequence",
        ):
            return self.tensor_converter.convert_sequence(
                encoded_sequence
            )

        return encoded_sequence

    # --------------------------------------------------
    # Utility Methods
    # --------------------------------------------------

    def get_feature_dimension(
        self,
    ) -> int:
        return self.FEATURE_DIMENSION

    def get_protocol_mapping(
        self,
    ) -> Dict[str, int]:
        return dict(
            self.config.protocol_mapping
        )