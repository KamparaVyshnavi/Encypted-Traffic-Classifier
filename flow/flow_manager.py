from typing import Any, Dict, List, Optional

from flow.flow_table import FlowKey, FlowRecord, FlowTable


class FlowManager:
    """
    Flow module orchestrator.

    Responsibilities:
    - receive parsed packet
    - validate packet
    - build normalized flat 5-tuple flow key
    - update FlowTable
    - optionally connect to flow_timeout.py
    - optionally connect to sequence_builder.py
    """

    REQUIRED_FIELDS = (
        "timestamp",
        "src_ip",
        "dst_ip",
        "src_port",
        "dst_port",
        "protocol",
        "packet_len",
    )

    def __init__(
        self,
        flow_table: Optional[FlowTable] = None,
        timeout_handler: Optional[Any] = None,
        sequence_builder: Optional[Any] = None,
    ) -> None:
        self.flow_table = flow_table or FlowTable()
        self.timeout_handler = timeout_handler
        self.sequence_builder = sequence_builder

    def process_packet(self, packet: Dict[str, Any]) -> Dict[str, Any]:
        if not self._is_valid_packet(packet):
            return {
                "status": "invalid_packet",
                "flow_key": None,
                "flow": None,
                "sequences": [],
                "expired_flows": [],
            }

        flow_key = self.build_flow_key(packet)
        flow = self.flow_table.add_packet(flow_key, packet)

        sequences = self._try_build_sequences(flow)
        expired_flows = self._try_expire_flows(packet["timestamp"])

        return {
            "status": "processed",
            "flow_key": flow_key,
            "flow": flow,
            "sequences": sequences,
            "expired_flows": expired_flows,
        }

    def build_flow_key(self, packet: Dict[str, Any]) -> FlowKey:
        """
        Creates normalized bidirectional flat 5-tuple.

        Output:
            (
                ip1,
                ip2,
                port1,
                port2,
                protocol
            )

        Forward and reverse packets produce the same key.
        """

        endpoint_a = (
            str(packet["src_ip"]),
            int(packet["src_port"]),
        )

        endpoint_b = (
            str(packet["dst_ip"]),
            int(packet["dst_port"]),
        )

        protocol = str(packet["protocol"]).upper()

        endpoint_1, endpoint_2 = sorted([endpoint_a, endpoint_b])

        ip1, port1 = endpoint_1
        ip2, port2 = endpoint_2

        return ip1, ip2, port1, port2, protocol

    def _is_valid_packet(self, packet: Dict[str, Any]) -> bool:
        for field in self.REQUIRED_FIELDS:
            if field not in packet:
                return False

        try:
            float(packet["timestamp"])
            int(packet["src_port"])
            int(packet["dst_port"])
            int(packet["packet_len"])
        except (TypeError, ValueError):
            return False

        if not packet["src_ip"] or not packet["dst_ip"]:
            return False

        if not packet["protocol"]:
            return False

        return True

    def _try_build_sequences(self, flow: FlowRecord) -> List[Any]:
        if self.sequence_builder is None:
            return []

        if hasattr(self.sequence_builder, "build_ready_sequences"):
            return self.sequence_builder.build_ready_sequences(flow)

        return []

    def _try_expire_flows(self, current_time: float) -> List[FlowRecord]:
        if self.timeout_handler is None:
            return []

        active_flows = self.flow_table.get_all_flows()

        if hasattr(self.timeout_handler, "find_expired_flows"):
            expired_flows = self.timeout_handler.find_expired_flows(
                active_flows=active_flows,
                current_time=float(current_time),
            )

            for flow in expired_flows:
                self.flow_table.remove_flow(flow.flow_key)

            return expired_flows

        return []

    def get_flow(self, flow_key: FlowKey) -> Optional[FlowRecord]:
        return self.flow_table.get_flow(flow_key)

    def get_all_flows(self) -> List[FlowRecord]:
        return self.flow_table.get_all_flows()

    def flow_count(self) -> int:
        return self.flow_table.flow_count()

    def clear(self) -> None:
        self.flow_table.clear()