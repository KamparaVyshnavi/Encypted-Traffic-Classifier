# preprocessing/handshake_detector.py


# Handshake detector serves two purposes:
#
# 1. Detect complete TCP/TLS handshakes for handshake analysis.
#    - TCP  : SYN -> SYNACK -> ACK
#    - TLS  : ClientHello -> ServerHello
#
# 2. Extract baseline latency for Novelty-1 even when a complete
#    handshake is not available.
#    - TCP baseline requires only SYN + SYNACK.
#    - TLS baseline requires only ClientHello + ServerHello.
#
# Baseline extraction priority:
#     TCP Baseline
#         ↓
#     TLS Baseline
#         ↓
#     Proxy Baseline
#
# Therefore, if both TCP and TLS baselines are available in a flow,
# the TCP baseline is used and TLS baseline is ignored.


from typing import Dict, List


class HandshakeDetector:
    """
    Detects TCP/TLS handshakes and extracts handshake metadata.

    Integration Hooks
    -----------------
    Backward:
        - packet_parser.py
        - flow_manager.py
        - sequence_builder.py

    Forward:
        - feature_encoder.py
        - normalizer.py
        - temporal_cnn.py
        - multi_exit.py
    """

    def __init__(
        self,
        feature_encoder=None,
        normalizer=None
    ):
        self.feature_encoder = feature_encoder
        self.normalizer = normalizer

    # ==========================================================
    # Public API
    # ==========================================================

    def process_flow(self, flow):
        """
        Process a single flow and identify handshake packets.

        Returns:
            dict
        """

        tcp_packets = self.detect_tcp_handshake(flow)
        tls_packets = self.detect_tls_handshake(flow)
        baseline_info = self.extract_baseline_latency(flow)

        handshake_packets = []

        seen = set()

        for pkt in tcp_packets + tls_packets:

            packet_id = id(pkt)

            if packet_id not in seen:
                handshake_packets.append(pkt)
                seen.add(packet_id)

        handshake_packets.sort(
            key=lambda p: p["timestamp"]
        )

        boundary_index = self.find_handshake_boundary(
            flow,
            handshake_packets
        )

        features = self.extract_handshake_features(
            flow,
            handshake_packets
        )

        result = {
            "handshake_detected": len(handshake_packets) > 0,
            "tcp_handshake_packets": tcp_packets,
            "tls_handshake_packets": tls_packets,
            "handshake_packets": handshake_packets,
            "data_packets": flow.packets[boundary_index:],
            "boundary_index": boundary_index,
            "handshake_features": features,
            "baseline_info": baseline_info,
        }

        self._attach_to_flow(
            flow,
            result
        )

        return result

    # ==========================================================
    # TCP Handshake Detection
    # ==========================================================

    def detect_tcp_handshake(
        self,
        flow
    ) -> List[Dict]:
        """
        Detect TCP 3-way handshake:

            SYN
            SYN-ACK
            ACK
        """

        syn_pkt = None
        syn_ack_pkt = None
        ack_pkt = None

        for pkt in flow.packets:

            if pkt["protocol"] != "TCP":
                continue

            flags = pkt["tcp_flags"]

            is_syn = bool(flags & 0x02)
            is_ack = bool(flags & 0x10)

            # SYN
            if syn_pkt is None:

                if is_syn and not is_ack:
                    syn_pkt = pkt
                    continue

            # SYN-ACK
            if syn_pkt and syn_ack_pkt is None:

                if is_syn and is_ack:
                    syn_ack_pkt = pkt
                    continue

            # Final ACK
            if syn_pkt and syn_ack_pkt and ack_pkt is None:

                if is_ack and not is_syn:
                    ack_pkt = pkt
                    break

        if (
            syn_pkt is not None
            and syn_ack_pkt is not None
            and ack_pkt is not None
        ):
            return [
                syn_pkt,
                syn_ack_pkt,
                ack_pkt
            ]

        return []

    # ==========================================================
    # TLS Handshake Detection
    # ==========================================================

    def detect_tls_handshake(
    self,
    flow
) -> List[Dict]:

        handshake_packets = []

        found_client_hello = False
        found_server_hello = False

        for pkt in flow.packets:

            if pkt.get("tls_record_type") != "handshake":
                continue

            handshake_type = pkt.get(
                "tls_handshake_type"
            )

            if handshake_type == "client_hello":
                found_client_hello = True

            elif handshake_type == "server_hello":
                found_server_hello = True

            handshake_packets.append(pkt)

        if found_client_hello and found_server_hello:
            return handshake_packets

        return []
    
    # ==========================================================
# Baseline Latency Extraction
# ==========================================================

    def extract_baseline_latency(
        self,
        flow
    ) -> Dict:
        """
        Novelty-1 baseline extraction.

        Priority:
            TCP SYN -> SYNACK
            TLS ClientHello -> ServerHello
            Proxy baseline
        """

        tcp_baseline = self._extract_tcp_baseline(
            flow
        )

        if tcp_baseline["baseline_available"]:
            return tcp_baseline

        tls_baseline = self._extract_tls_baseline(
            flow
        )

        if tls_baseline["baseline_available"]:
            return tls_baseline

        return self._extract_proxy_baseline(
            flow
        )


    def _extract_tcp_baseline(
        self,
        flow
    ) -> Dict:

        syn_pkt = None
        syn_ack_pkt = None

        for pkt in flow.packets:

            if pkt["protocol"] != "TCP":
                continue

            flags = pkt["tcp_flags"]

            is_syn = bool(flags & 0x02)
            is_ack = bool(flags & 0x10)

            if syn_pkt is None:

                if is_syn and not is_ack:
                    syn_pkt = pkt
                    continue

            if syn_pkt is not None:

                if is_syn and is_ack:
                    syn_ack_pkt = pkt
                    break

        if syn_pkt and syn_ack_pkt:

            latency = (
                float(syn_ack_pkt["timestamp"])
                - float(syn_pkt["timestamp"])
            )

            if latency > 0:

                return {
                    "baseline_available": True,
                    "baseline_type": "tcp",
                    "baseline_latency": latency,
                }

        return {
            "baseline_available": False
        }


    def _extract_tls_baseline(
        self,
        flow
    ) -> Dict:

        client_hello = None
        server_hello = None

        for pkt in flow.packets:

            handshake_type = pkt.get(
                "tls_handshake_type"
            )

            if handshake_type is None:
                continue

            if (
                handshake_type == "client_hello"
                and client_hello is None
            ):
                client_hello = pkt
                continue

            if (
                handshake_type == "server_hello"
                and client_hello is not None
            ):
                server_hello = pkt
                break

        if client_hello and server_hello:

            latency = (
                float(server_hello["timestamp"])
                - float(client_hello["timestamp"])
            )

            if latency > 0:

                return {
                    "baseline_available": True,
                    "baseline_type": "tls",
                    "baseline_latency": latency,
                }

        return {
            "baseline_available": False
        }


    def _extract_proxy_baseline(
    self,
    flow
) -> Dict:

        if len(flow.packets) < 5:

            return {
                "baseline_available": False
            }

        timestamps = []

        for pkt in flow.packets[:10]:

            timestamps.append(
                float(pkt["timestamp"])
            )

        gaps = []

        for i in range(
            1,
            len(timestamps)
        ):

            gap = (
                timestamps[i]
                - timestamps[i - 1]
            )

            if gap > 0:
                gaps.append(gap)

        if not gaps:

            return {
                "baseline_available": False
            }

        gaps.sort()

        median_gap = gaps[
            len(gaps) // 2
        ]

        return {
            "baseline_available": True,
            "baseline_type": "proxy",
            "baseline_latency": median_gap,
        }
    # ==========================================================
    # Handshake Boundary
    # ==========================================================

    def find_handshake_boundary(
        self,
        flow,
        handshake_packets
    ) -> int:
        """
        Return index immediately after the last
        handshake packet.
        """

        if not handshake_packets:
            return 0

        last_packet = handshake_packets[-1]

        try:
            return (
                flow.packets.index(last_packet)
                + 1
            )

        except ValueError:
            return 0

    # ==========================================================
    # Feature Extraction
    # ==========================================================

    def extract_handshake_features(
        self,
        flow,
        handshake_packets
    ) -> Dict:
        """
        Extract handshake metadata.
        """

        if not handshake_packets:

            return {
                "handshake_packet_count": 0,
                "handshake_duration": 0.0,
                "handshake_bytes": 0,
                "client_to_server": 0,
                "server_to_client": 0,
            }

        first_time = float(
            handshake_packets[0]["timestamp"]
        )

        last_time = float(
            handshake_packets[-1]["timestamp"]
        )

        total_bytes = 0

        client_to_server = 0
        server_to_client = 0

        # SYN sender is treated as client
        client_ip = handshake_packets[0]["src_ip"]

        for pkt in handshake_packets:

            total_bytes += pkt["packet_len"]

            if pkt["src_ip"] == client_ip:
                client_to_server += 1
            else:
                server_to_client += 1

        return {
            "handshake_packet_count":
                len(handshake_packets),

            "handshake_duration":
                max(
                    0.0,
                    last_time - first_time
                ),

            "handshake_bytes":
                total_bytes,

            "client_to_server":
                client_to_server,

            "server_to_client":
                server_to_client,
        }

    # ==========================================================
    # Integration Helpers
    # ==========================================================

    def _attach_to_flow(
        self,
        flow,
        result
    ):
        """
        Attach handshake information to flow.
        """

        flow.handshake_detected = result[
            "handshake_detected"
        ]

        flow.tcp_handshake_packets = result[
            "tcp_handshake_packets"
        ]

        flow.tls_handshake_packets = result[
            "tls_handshake_packets"
        ]

        flow.handshake_packets = result[
            "handshake_packets"
        ]

        flow.handshake_features = result[
            "handshake_features"
        ]

        flow.handshake_boundary_index = result[
            "boundary_index"
        ]
        flow.baseline_info = result[
    "baseline_info"
]

        flow.baseline_latency = (
            result["baseline_info"].get(
                "baseline_latency"
            )
        )

        flow.baseline_type = (
            result["baseline_info"].get(
                "baseline_type"
            )
        )

        flow.baseline_available = (
    result["baseline_info"].get(
        "baseline_available",
        False
    )
)