# from scapy.all import rdpcap

# from scapy.layers.tls.record import TLS

# pcap_file = (
#     r"datasets/raw_pcaps/iscx_official/"
#     r"VPN-PCAPs-01/vpn_hangouts_audio2.pcap"
# )

# packets = rdpcap(pcap_file)

# count = 0

# for pkt in packets:

#     if TLS not in pkt:
#         continue

#     tls = pkt[TLS]

#     try:

#         if tls.type == 22:

#             count += 1

#             print("=" * 60)
#             print(pkt.summary())

#             tls.show()

#             print()

#             if count >= 5:
#                 break

#     except Exception:
#         pass

# print()
# print(f"Handshake Records Found: {count}")


from scapy.all import rdpcap

from capture.packet_parser import PacketParser


pcap_file = (
    r"datasets/raw_pcaps/iscx_official/VPN-PCAPs-01/vpn_hangouts_audio2.pcap"
)


def main():

    parser = PacketParser()

    packets = rdpcap(pcap_file)

    client_hello_count = 0
    server_hello_count = 0
    certificate_count = 0
    client_key_exchange_count = 0
    finished_count = 0

    tls_packet_count = 0

    sample_packets = []

    print("=" * 80)
    print("TLS PARSER VALIDATION")
    print("=" * 80)

    for raw_packet in packets:

        parsed_packet = parser.parse_packet(raw_packet)

        if parsed_packet is None:
            continue

        handshake_type = parsed_packet.get(
            "tls_handshake_type"
        )

        record_type = parsed_packet.get(
            "tls_record_type"
        )

        if (
            record_type is not None
            or handshake_type is not None
        ):
            tls_packet_count += 1

        if handshake_type == "client_hello":
            client_hello_count += 1

        elif handshake_type == "server_hello":
            server_hello_count += 1

        elif handshake_type == "certificate":
            certificate_count += 1

        elif handshake_type == "client_key_exchange":
            client_key_exchange_count += 1

        elif handshake_type == "finished":
            finished_count += 1

        if (
            handshake_type is not None
            and len(sample_packets) < 10
        ):
            sample_packets.append(parsed_packet)

    print()
    print("=" * 80)
    print("TLS COUNTS")
    print("=" * 80)

    print(f"TLS Packets                : {tls_packet_count}")
    print(f"Client Hello              : {client_hello_count}")
    print(f"Server Hello              : {server_hello_count}")
    print(f"Certificate               : {certificate_count}")
    print(f"Client Key Exchange       : {client_key_exchange_count}")
    print(f"Finished                  : {finished_count}")

    print()
    print("=" * 80)
    print("SAMPLE TLS PACKETS")
    print("=" * 80)

    for idx, packet in enumerate(
        sample_packets,
        start=1
    ):

        print(f"\nSample {idx}")

        print(
            f"Source      : "
            f"{packet['src_ip']}:{packet['src_port']}"
        )

        print(
            f"Destination : "
            f"{packet['dst_ip']}:{packet['dst_port']}"
        )

        print(
            f"Record Type : "
            f"{packet['tls_record_type']}"
        )

        print(
            f"Handshake   : "
            f"{packet['tls_handshake_type']}"
        )

        print(
            f"Version     : "
            f"{packet['tls_version']}"
        )


if __name__ == "__main__":
    main()