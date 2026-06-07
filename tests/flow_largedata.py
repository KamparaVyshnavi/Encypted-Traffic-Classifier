# tests/test_large_scale_flow_pipeline.py

from scapy.all import PcapReader

from capture.packet_parser import PacketParser
from flow.flow_manager import FlowManager


PCAP_FILE = "datasets/raw_pcaps/USTC_TFC/Benign/Outlook.pcap"


def main():

    parser = PacketParser()
    manager = FlowManager()

    packet_counter = 0

    with PcapReader(PCAP_FILE) as reader:

        for packet in reader:

            parsed = parser.parse_packet(packet)

            if parsed is None:
                continue

            result = manager.process_packet(parsed)

            if result["status"] != "processed":
                continue

            packet_counter += 1

    flows = manager.get_all_flows()

    print(f"Processed Packets: {packet_counter}")
    print(f"Total Flows: {manager.flow_count()}")

    print()
    print("Top 10 Flows By Packet Count")

    top_packet_flows = sorted(
        flows,
        key=lambda x: x.packet_count,
        reverse=True
    )[:10]

    for flow in top_packet_flows:
        print(
            flow.flow_key,
            flow.packet_count,
            flow.total_bytes
        )

    print()
    print("Validation")

    packet_sum = sum(f.packet_count for f in flows)

    print(
        "Packet Counter:",
        packet_counter
    )

    print(
        "Packet Sum Across Flows:",
        packet_sum
    )

    print(
        "Match:",
        packet_counter == packet_sum
    )


if __name__ == "__main__":
    main()