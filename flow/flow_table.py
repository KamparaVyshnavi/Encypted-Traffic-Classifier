from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


FlowKey = Tuple[str, str, int, int, str]
# Format:
# (
#   ip1,
#   ip2,
#   port1,
#   port2,
#   protocol
# )


@dataclass
class FlowRecord:
    flow_key: FlowKey
    packets: List[Dict[str, Any]] = field(default_factory=list)
    start_time: Optional[float] = None
    last_seen: Optional[float] = None
    packet_count: int = 0
    total_bytes: int = 0
    is_closed: bool = False

    def add_packet(self, packet: Dict[str, Any]) -> None:
        timestamp = float(packet["timestamp"])
        packet_len = int(packet.get("packet_len", 0))

        if self.start_time is None:
            self.start_time = timestamp

        self.last_seen = timestamp
        self.packet_count += 1
        self.total_bytes += packet_len
        self.packets.append(packet)

    def close(self) -> None:
        self.is_closed = True


class FlowTable:
    """
    Storage layer for active flows.

    It does not create flow keys.
    It only stores, updates, returns, and removes FlowRecord objects.
    """

    def __init__(self) -> None:
        self._active_flows: Dict[FlowKey, FlowRecord] = {}

    def has_flow(self, flow_key: FlowKey) -> bool:
        return flow_key in self._active_flows

    def create_flow(self, flow_key: FlowKey) -> FlowRecord:
        flow = FlowRecord(flow_key=flow_key)
        self._active_flows[flow_key] = flow
        return flow

    def add_packet(self, flow_key: FlowKey, packet: Dict[str, Any]) -> FlowRecord:
        if not self.has_flow(flow_key):
            self.create_flow(flow_key)

        flow = self._active_flows[flow_key]
        flow.add_packet(packet)
        return flow

    def get_flow(self, flow_key: FlowKey) -> Optional[FlowRecord]:
        return self._active_flows.get(flow_key)

    def get_all_flows(self) -> List[FlowRecord]:
        return list(self._active_flows.values())

    def remove_flow(self, flow_key: FlowKey) -> Optional[FlowRecord]:
        return self._active_flows.pop(flow_key, None)

    def close_flow(self, flow_key: FlowKey) -> Optional[FlowRecord]:
        flow = self.get_flow(flow_key)
        if flow:
            flow.close()
        return flow

    def clear(self) -> None:
        self._active_flows.clear()

    def flow_count(self) -> int:
        return len(self._active_flows)