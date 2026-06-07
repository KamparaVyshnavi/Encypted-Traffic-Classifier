from scapy.layers.inet import IP, TCP, UDP


class PacketParser:
    """
    Converts raw Scapy packets into a standardized format
    used by the rest of the pipeline.
    """

    def parse_packet(self, packet):
        """
        Parse a single Scapy packet.

        Returns:
            dict containing extracted packet information
            OR None if packet is unsupported.
        """

        # Ignore non-IP packets
        try:
            if IP not in packet:
                return None

            parsed_packet = {
                "timestamp": packet.time,
                "src_ip": packet[IP].src,
                "dst_ip": packet[IP].dst,
                "src_port": None,
                "dst_port": None,
                "protocol": None,
                "packet_len": len(packet)
            }

            # TCP Packet
            if TCP in packet:
                parsed_packet["protocol"] = "TCP"
                parsed_packet["src_port"] = packet[TCP].sport
                parsed_packet["dst_port"] = packet[TCP].dport

            # UDP Packet
            elif UDP in packet:
                parsed_packet["protocol"] = "UDP"
                parsed_packet["src_port"] = packet[UDP].sport
                parsed_packet["dst_port"] = packet[UDP].dport

            # Other IP protocol
            else:
                parsed_packet["protocol"] = str(packet[IP].proto)

            return parsed_packet
        except Exception as e:
            print(f"Packet parsing error:{e}")
            return None