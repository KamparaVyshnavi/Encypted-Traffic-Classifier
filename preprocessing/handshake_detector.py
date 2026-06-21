# # preprocessing/handshake_detector.py

# from typing import Dict, List, Optional


# class HandshakeDetector:
#     """
#     Detects TCP/TLS handshakes and extracts handshake metadata.

#     Integration Hooks:
#     ------------------
#     Backward:
#         - packet_parser.py
#         - flow_manager.py
#         - sequence_builder.py

#     Forward:
#         - feature_encoder.py
#         - normalizer.py
#         - temporal_cnn.py
#         - multi_exit.py
#     """

#     def __init__(
#         self,
#         feature_encoder=None,
#         normalizer=None
#     ):
#         self.feature_encoder = feature_encoder
#         self.normalizer = normalizer

#     # ==========================================================
#     # Public API
#     # ==========================================================

#     def process_flow(self, flow):
#         """
#         Main entry point.

#         Parameters
#         ----------
#         flow : Flow

#         Returns
#         -------
#         dict
#         """

#         tcp_packets = self.detect_tcp_handshake(flow)
#         tls_packets = self.detect_tls_handshake(flow)

#         handshake_packets = []

#         seen = set()

#         for pkt in tcp_packets + tls_packets:
#             packet_id = id(pkt)

#             if packet_id not in seen:
#                 handshake_packets.append(pkt)
#                 seen.add(packet_id)

#         handshake_packets.sort(
#             key=lambda p: getattr(p, "timestamp", 0)
#         )

#         boundary_index = self.find_handshake_boundary(
#             flow,
#             handshake_packets
#         )

#         features = self.extract_handshake_features(
#             flow,
#             handshake_packets
#         )

#         result = {
#             "handshake_detected": len(handshake_packets) > 0,
#             "tcp_handshake_packets": tcp_packets,
#             "tls_handshake_packets": tls_packets,
#             "handshake_packets": handshake_packets,
#             "data_packets": flow.packets[boundary_index:],
#             "boundary_index": boundary_index,
#             "handshake_features": features
#         }

#         self._attach_to_flow(flow, result)

#         return result

#     # ==========================================================
#     # TCP Handshake Detection
#     # ==========================================================

#     def detect_tcp_handshake(self, flow) -> List:
#         """
#         Detect TCP 3-way handshake:

#             SYN
#             SYN-ACK
#             ACK
#         """

#         syn_pkt = None
#         syn_ack_pkt = None
#         ack_pkt = None

#         for pkt in flow.packets:

#             if pkt["protocol"] != "TCP":
#                 continue

#             flags = pkt["tcp_flags"]

#             is_syn = bool(flags & 0x02)
#             is_ack = bool(flags & 0x10)

#             # SYN
#             if syn_pkt is None:

#                 if is_syn and not is_ack:
#                     syn_pkt = pkt
#                     continue

#             # SYN-ACK
#             if syn_pkt and syn_ack_pkt is None:

#                 if is_syn and is_ack:
#                     syn_ack_pkt = pkt
#                     continue

#             # Final ACK
#             if syn_pkt and syn_ack_pkt and ack_pkt is None:

#                 if is_ack and not is_syn:
#                     ack_pkt = pkt
#                     break

#         if syn_pkt and syn_ack_pkt and ack_pkt:

#             return [
#                 syn_pkt,
#                 syn_ack_pkt,
#                 ack_pkt
#             ]

#         return []
#     # ==========================================================
#     # TLS Handshake Detection
#     # ==========================================================

#     def detect_tls_handshake(self, flow) -> List:
#         """
#         Detect TLS handshake records.

#         Expected parser fields:

#             packet.protocol
#             packet.tls_type
#             packet.tls_handshake_type
#         """

#         handshake_packets = []

#         tls_handshake_types = {
#             "client_hello",
#             "server_hello",
#             "certificate",
#             "server_key_exchange",
#             "client_key_exchange",
#             "finished"
#         }

#         for pkt in flow.packets:

#             protocol = getattr(pkt, "protocol", "").lower()

#             if protocol not in ("tls", "ssl"):
#                 continue

#             handshake_type = getattr(
#                 pkt,
#                 "tls_handshake_type",
#                 ""
#             ).lower()

#             if handshake_type in tls_handshake_types:
#                 handshake_packets.append(pkt)

#         return handshake_packets

#     # ==========================================================
#     # Boundary Detection
#     # ==========================================================

#     def find_handshake_boundary(
#         self,
#         flow,
#         handshake_packets
#     ) -> int:
#         """
#         Returns first packet index after handshake.
#         """

#         if not handshake_packets:
#             return 0

#         last_packet = handshake_packets[-1]

#         try:
#             return flow.packets.index(last_packet) + 1
#         except ValueError:
#             return 0

#     # ==========================================================
#     # Feature Extraction
#     # ==========================================================

#     def extract_handshake_features(
#         self,
#         flow,
#         handshake_packets
#     ) -> Dict:
#         """
#         Extract metadata from handshake.
#         """

#         if not handshake_packets:

#             return {
#                 "handshake_packet_count": 0,
#                 "handshake_duration": 0.0,
#                 "handshake_bytes": 0,
#                 "client_to_server": 0,
#                 "server_to_client": 0
#             }

#         first_time = getattr(
#             handshake_packets[0],
#             "timestamp",
#             0
#         )

#         last_time = getattr(
#             handshake_packets[-1],
#             "timestamp",
#             0
#         )

#         total_bytes = 0

#         c2s = 0
#         s2c = 0

#         client_ip = getattr(
#             flow,
#             "src_ip",
#             None
#         )

#         for pkt in handshake_packets:

#             total_bytes += getattr(pkt, "length", 0)

#             src_ip = getattr(pkt, "src_ip", None)

#             if src_ip == client_ip:
#                 c2s += 1
#             else:
#                 s2c += 1

#         return {
#             "handshake_packet_count":
#                 len(handshake_packets),

#             "handshake_duration":
#                 max(0.0, last_time - first_time),

#             "handshake_bytes":
#                 total_bytes,

#             "client_to_server":
#                 c2s,

#             "server_to_client":
#                 s2c
#         }

#     # ==========================================================
#     # Integration Helpers
#     # ==========================================================

#     def _attach_to_flow(
#         self,
#         flow,
#         result
#     ):
#         """
#         Attach results directly to flow object.
#         """

#         flow.handshake_detected = result[
#             "handshake_detected"
#         ]

#         flow.tcp_handshake_packets = result[
#             "tcp_handshake_packets"
#         ]

#         flow.tls_handshake_packets = result[
#             "tls_handshake_packets"
#         ]

#         flow.handshake_packets = result[
#             "handshake_packets"
#         ]

#         flow.handshake_features = result[
#             "handshake_features"
#         ]

#         flow.handshake_boundary_index = result[
#             "boundary_index"
#         ]

#     # ==========================================================
#     # Utility Methods
#     # ==========================================================

#     def _get_tcp_flags(self, packet):

#         return packet.get(
#             "tcp_flags",
#             0
#         )



# preprocessing/handshake_detector.py

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
        """
        Detect TLS handshake packets.

        NOTE:
        Current PacketParser does not expose TLS metadata.

        This method is retained for future integration
        when tls_handshake_type becomes available.
        """

        handshake_packets = []

        tls_handshake_types = {
            "client_hello",
            "server_hello",
            "certificate",
            "server_key_exchange",
            "client_key_exchange",
            "finished",
        }

        for pkt in flow.packets:

            protocol = pkt.get(
                "protocol",
                ""
            ).lower()

            if protocol not in (
                "tls",
                "ssl"
            ):
                continue

            handshake_type = pkt.get(
                "tls_handshake_type",
                ""
            ).lower()

            if handshake_type in tls_handshake_types:
                handshake_packets.append(pkt)

        return handshake_packets

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