from scapy.all import PcapReader

from capture.packet_parser import PacketParser
from flow.flow_manager import FlowManager


PCAP_FILE = "datasets/raw_pcaps/USTC_TFC/Benign/WorldOfWarcraft.pcap"


def main():

    parser = PacketParser()
    manager = FlowManager()

    total_packets = 0

    with PcapReader(PCAP_FILE) as reader:

        for packet in reader:

            parsed_packet = parser.parse_packet(packet)

            if parsed_packet is None:
                continue

            result = manager.process_packet(parsed_packet)

            if result["status"] != "processed":
                continue

            total_packets += 1

    flows = manager.get_all_flows()

    print("\n========== BASIC STATISTICS ==========\n")

    print("Processed Packets :", total_packets)
    print("Total Flows       :", len(flows))

    if not flows:
        print("No flows found.")
        return

    print("\n========== FLOW SIZE DISTRIBUTION ==========\n")

    distribution = {}

    for flow in flows:

        distribution[flow.packet_count] = (
            distribution.get(flow.packet_count, 0) + 1
        )

    for size in sorted(distribution):

        print(
            f"{size:>3} packets -> "
            f"{distribution[size]} flows"
        )

    largest_flow = max(
        flows,
        key=lambda flow: flow.packet_count
    )

    print("\n========== LARGEST FLOW ==========\n")

    print("Flow Key:")
    print(largest_flow.flow_key)

    print()

    print("Packet Count:")
    print(largest_flow.packet_count)

    print()

    print("Start Time:")
    print(largest_flow.start_time)

    print()

    print("Last Seen:")
    print(largest_flow.last_seen)

    print()

    print(
        "Flow Duration:"
    )

    print(
        largest_flow.last_seen
        - largest_flow.start_time
    )

    print("\n========== PACKETS INSIDE LARGEST FLOW ==========\n")

    for index, packet in enumerate(
        largest_flow.packets,
        start=1
    ):

        print(
            f"{index:>2}",
            packet["timestamp"],
            packet["src_ip"],
            packet["src_port"],
            "->",
            packet["dst_ip"],
            packet["dst_port"],
            packet["protocol"],
            packet["packet_len"]
        )

    print("\n========== DIRECTION CHECK ==========\n")

    for packet in largest_flow.packets:

        print(
            packet["src_ip"],
            packet["src_port"],
            "->",
            packet["dst_ip"],
            packet["dst_port"]
        )

    ip1 = largest_flow.flow_key[0]
    ip2 = largest_flow.flow_key[1]

    print("\n========== RELATED FLOWS ==========\n")

    related_count = 0

    for flow in flows:

        f_ip1 = flow.flow_key[0]
        f_ip2 = flow.flow_key[1]

        if {ip1, ip2} == {f_ip1, f_ip2}:

            print(
                flow.flow_key,
                "Packets:",
                flow.packet_count
            )

            related_count += 1

    print()

    print(
        "Related Flow Count:",
        related_count
    )


if __name__ == "__main__":
    main()