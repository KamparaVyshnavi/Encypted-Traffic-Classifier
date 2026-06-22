# from scapy.layers.inet import IP, TCP, UDP
# from scapy.layers.tls.record import TLS


# class PacketParser:
#     """
#     Converts raw Scapy packets into a standardized format
#     used by the rest of the pipeline.
#     """

#     def parse_packet(self, packet):
#         """
#         Parse a single Scapy packet.

#         Returns:
#             dict containing extracted packet information
#             OR None if packet is unsupported.
#         """

#         # Ignore non-IP packets
#         try:
#             if IP not in packet:
#                 return None

#             parsed_packet = {
#                 "timestamp": packet.time,
#                 "src_ip": packet[IP].src,
#                 "dst_ip": packet[IP].dst,
#                 "src_port": None,
#                 "dst_port": None,
#                 "protocol": None,
#                 "packet_len": len(packet),

#                 # Future feature extraction hooks
#                 "tcp_flags": None,
#                 # TLS metadata
#                 "tls_record_type": None,
#                 "tls_handshake_type": None,
#                 "tls_version": None,
#             }

#             # TCP Packet
#             # TCP Packet
#             if TCP in packet:
#                 parsed_packet["protocol"] = "TCP"
#                 parsed_packet["src_port"] = packet[TCP].sport
#                 parsed_packet["dst_port"] = packet[TCP].dport

#                 # Numeric TCP flags value
#                 parsed_packet["tcp_flags"] = int(packet[TCP].flags)

#             # UDP Packet
#             elif UDP in packet:
#                 parsed_packet["protocol"] = "UDP"
#                 parsed_packet["src_port"] = packet[UDP].sport
#                 parsed_packet["dst_port"] = packet[UDP].dport

#             # Other IP protocol
#             else:
#                 return None

#             # TLS Metadata Extraction
#             self._extract_tls_metadata(
#                 packet,
#                 parsed_packet
#             )

#             return parsed_packet

#         except Exception as e:
#             print(f"Packet parsing error: {e}")
#             return 
    
#     def _extract_tls_metadata(
#     self,
#     packet,
#     parsed_packet
# ):
#         """
#         Extract TLS metadata if present.
#         """

#         if TLS not in packet:
#             return

#         try:

#             tls_layer = packet[TLS]

#             parsed_packet["tls_record_type"] = str(
#                 tls_layer.type
#             ).lower()

#             parsed_packet["tls_version"] = str(
#                 tls_layer.version
#             )

#             if hasattr(tls_layer, "msg"):

#                 for msg in tls_layer.msg:

#                     if hasattr(msg, "msgtype"):

#                         parsed_packet[
#                             "tls_handshake_type"
#                         ] = str(
#                             msg.msgtype
#                         ).lower()

#                         break

#         except Exception:
#             pass



from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.tls.record import TLS


class PacketParser:
    """
    Converts raw Scapy packets into a standardized format
    used by the rest of the pipeline.
    """

    TLS_RECORD_TYPES = {
        20: "change_cipher_spec",
        21: "alert",
        22: "handshake",
        23: "application_data",
    }

    TLS_HANDSHAKE_TYPES = {
        1: "client_hello",
        2: "server_hello",
        4: "new_session_ticket",
        11: "certificate",
        12: "server_key_exchange",
        14: "server_hello_done",
        16: "client_key_exchange",
        20: "finished",
    }

    TLS_VERSIONS = {
        768: "SSL3.0",
        769: "TLS1.0",
        770: "TLS1.1",
        771: "TLS1.2",
        772: "TLS1.3",
    }

    def parse_packet(self, packet):
        """
        Parse a single Scapy packet.

        Returns:
            dict containing extracted packet information
            OR None if packet is unsupported.
        """

        try:

            # Ignore non-IP packets
            if IP not in packet:
                return None

            parsed_packet = {
                "timestamp": packet.time,
                "src_ip": packet[IP].src,
                "dst_ip": packet[IP].dst,
                "src_port": None,
                "dst_port": None,
                "protocol": None,
                "packet_len": len(packet),

                # TCP metadata
                "tcp_flags": None,

                # TLS metadata
                "tls_record_type": None,
                "tls_handshake_type": None,
                "tls_version": None,
            }

            # TCP Packet
            if TCP in packet:

                parsed_packet["protocol"] = "TCP"
                parsed_packet["src_port"] = packet[TCP].sport
                parsed_packet["dst_port"] = packet[TCP].dport

                # Numeric TCP flags value
                parsed_packet["tcp_flags"] = int(
                    packet[TCP].flags
                )

            # UDP Packet
            elif UDP in packet:

                parsed_packet["protocol"] = "UDP"
                parsed_packet["src_port"] = packet[UDP].sport
                parsed_packet["dst_port"] = packet[UDP].dport

            # Other IP protocol
            else:
                return None

            # TLS Metadata Extraction
            self._extract_tls_metadata(
                packet,
                parsed_packet
            )

            return parsed_packet

        except Exception as e:

            print(
                f"Packet parsing error: {e}"
            )

            return None

    def _extract_tls_metadata(
        self,
        packet,
        parsed_packet
    ):
        """
        Extract TLS metadata if present.
        """

        if TLS not in packet:
            return

        try:

            tls_layer = packet[TLS]

            # ----------------------------------
            # TLS Record Type
            # ----------------------------------

            record_type = getattr(
                tls_layer,
                "type",
                None
            )

            parsed_packet[
                "tls_record_type"
            ] = self.TLS_RECORD_TYPES.get(
                record_type,
                str(record_type)
                if record_type is not None
                else None
            )

            # ----------------------------------
            # TLS Version
            # ----------------------------------

            version = getattr(
                tls_layer,
                "version",
                None
            )

            parsed_packet[
                "tls_version"
            ] = self.TLS_VERSIONS.get(
                version,
                str(version)
                if version is not None
                else None
            )

            # ----------------------------------
            # TLS Handshake Type
            # ----------------------------------

            if hasattr(tls_layer, "msg"):

                for msg in tls_layer.msg:

                    if not hasattr(
                        msg,
                        "msgtype"
                    ):
                        continue

                    msg_type = msg.msgtype

                    parsed_packet[
                        "tls_handshake_type"
                    ] = self.TLS_HANDSHAKE_TYPES.get(
                        msg_type,
                        str(msg_type)
                    )

                    break

        except Exception:
            pass