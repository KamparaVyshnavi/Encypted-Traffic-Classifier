from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from flow.flow_table import FlowKey, FlowRecord


@dataclass
class SequenceRecord:
    flow_key: FlowKey
    sequence: List[Dict[str, Any]]
    original_packet_count: int
    sequence_length: int
    valid_exit_points: List[int]
    is_padded: bool
    is_closed: bool


@dataclass
class SequenceBuilderConfig:
    sequence_length: int = 20
    exit_points: Optional[List[int]] = None
    sort_by_timestamp: bool = True
    build_only_closed_flows: bool = False


class SequenceBuilder:
    """
    Builds fixed-length packet sequences from FlowRecord objects.

    Compatible with:
    - FlowManager
    - FlowTable
    - FlowRecord
    - future FeatureEncoder
    - future HandshakeDetector
    - future TensorConverter
    - future Multi-Exit CNN
    """

    def __init__(
        self,
        config: Optional[SequenceBuilderConfig] = None,
    ) -> None:

        self.config = config or SequenceBuilderConfig()

        if self.config.exit_points is None:
            self.config.exit_points = [5, 10, 15, 20]

        self.exit_points = sorted(self.config.exit_points)
        self.sequence_length = self.config.sequence_length
        self.min_packets = min(self.exit_points)

        if self.sequence_length < max(self.exit_points):
            raise ValueError(
                "sequence_length must be >= max(exit_points)"
            )

        # flow_key -> last emitted exit point
        self._last_emitted_exit: Dict[FlowKey, int] = {}

    def is_valid_flow(self, flow: FlowRecord) -> bool:
        if flow is None:
            return False

        if (
            self.config.build_only_closed_flows
            and not flow.is_closed
        ):
            return False

        return flow.packet_count >= self.min_packets

    def build_sequence(
        self,
        flow: FlowRecord,
    ) -> Optional[SequenceRecord]:

        if not self.is_valid_flow(flow):
            return None

        packets = list(flow.packets)

        if self.config.sort_by_timestamp:
            packets.sort(
                key=lambda packet:
                float(packet["timestamp"])
            )

        selected_packets = packets[:self.sequence_length]

        sequence = self.pad_sequence(
            selected_packets
        )

        return SequenceRecord(
            flow_key=flow.flow_key,
            sequence=sequence,
            original_packet_count=flow.packet_count,
            sequence_length=self.sequence_length,
            valid_exit_points=self._valid_exit_points(
                flow.packet_count
            ),
            is_padded=(
                len(selected_packets)
                < self.sequence_length
            ),
            is_closed=flow.is_closed,
        )

    def build_sequences(
        self,
        flows: List[FlowRecord],
    ) -> List[SequenceRecord]:

        sequences: List[SequenceRecord] = []

        for flow in flows:

            sequence_record = self.build_sequence(
                flow
            )

            if sequence_record is not None:
                sequences.append(
                    sequence_record
                )

        return sequences

    def build_from_flow_table(
        self,
        flow_table: Any,
    ) -> List[SequenceRecord]:

        return self.build_sequences(
            flow_table.get_all_flows()
        )

    def build_ready_sequences(
        self,
        flow: FlowRecord,
    ) -> List[SequenceRecord]:
        """
        Called by FlowManager after every packet.

        Emits a sequence only when a NEW exit
        point becomes available.

        Example:
            exit_points = [5,10,15,20]

            packet 5  -> emit
            packet 10 -> emit
            packet 15 -> emit
            packet 20 -> emit

            packet 6,7,8,9 -> no emit
        """

        if not self.is_valid_flow(flow):
            return []

        valid_exits = self._valid_exit_points(
            flow.packet_count
        )

        if not valid_exits:
            return []

        newest_exit = max(valid_exits)

        last_exit = self._last_emitted_exit.get(
            flow.flow_key,
            0
        )

        if newest_exit <= last_exit:
            return []

        sequence_record = self.build_sequence(
            flow
        )

        if sequence_record is None:
            return []

        self._last_emitted_exit[
            flow.flow_key
        ] = newest_exit

        return [sequence_record]

    def pad_sequence(
        self,
        packets: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:

        sequence = [
            self._normalize_packet(
                packet,
                is_padding=False
            )
            for packet in packets
        ]

        while len(sequence) < self.sequence_length:
            sequence.append(
                self._padding_packet()
            )

        return sequence

    def _valid_exit_points(
        self,
        packet_count: int
    ) -> List[int]:

        return [
            exit_point
            for exit_point in self.exit_points
            if packet_count >= exit_point
        ]

    def _normalize_packet(
        self,
        packet: Dict[str, Any],
        is_padding: bool,
    ) -> Dict[str, Any]:

        packet_len = int(
            packet.get("packet_len", 0)
        )

        return {
            "timestamp":
                float(
                    packet.get(
                        "timestamp",
                        0.0
                    )
                ),

            "src_ip":
                packet.get("src_ip"),

            "dst_ip":
                packet.get("dst_ip"),

            "src_port":
                int(
                    packet.get(
                        "src_port",
                        0
                    )
                ),

            "dst_port":
                int(
                    packet.get(
                        "dst_port",
                        0
                    )
                ),

            "protocol":
                str(
                    packet.get(
                        "protocol",
                        ""
                    )
                ).upper(),

            "packet_len":
                packet_len,

            "packet_length":
                packet_len,

            "direction":
                packet.get(
                    "direction"
                ),

            "tcp_flags":
                packet.get(
                    "tcp_flags"
                ),

            "payload_len":
                int(
                    packet.get(
                        "payload_len",
                        0
                    )
                ),

            "is_padding":
                is_padding,
        }

    def _padding_packet(
        self
    ) -> Dict[str, Any]:

        return {
            "timestamp": 0.0,
            "src_ip": None,
            "dst_ip": None,
            "src_port": 0,
            "dst_port": 0,
            "protocol": "PAD",
            "packet_len": 0,
            "packet_length": 0,
            "direction": None,
            "tcp_flags": None,
            "payload_len": 0,
            "is_padding": True,
        }