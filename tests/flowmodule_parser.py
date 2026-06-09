from scapy.all import PcapReader

from capture.packet_parser import PacketParser
from flow.flow_manager import FlowManager


PCAP_FILE = "datasets/raw_pcaps/USTC_TFC/Benign/WorldOfWarcraft.pcap"


def main():

    parser = PacketParser()
    manager = FlowManager()

    total_packets_seen = 0
    parsed_packets = 0
    processed_packets = 0

    with PcapReader(PCAP_FILE) as reader:

        for packet in reader:

            total_packets_seen += 1

            parsed_packet = parser.parse_packet(packet)

            if parsed_packet is None:
                continue

            parsed_packets += 1

            result = manager.process_packet(parsed_packet)

            if result["status"] != "processed":
                continue

            processed_packets += 1

    flows = manager.get_all_flows()

    print("\n========== FLOW SIZE ANALYSIS ==========\n")

    flow_size_distribution = {}

    for flow in flows:

        packet_count = flow.packet_count

        flow_size_distribution[packet_count] = (
            flow_size_distribution.get(packet_count, 0) + 1
        )

    for packet_count in sorted(flow_size_distribution):

        print(
            f"{packet_count:>3} packets -> "
            f"{flow_size_distribution[packet_count]} flows"
        )

    if flows:

        max_packets = max(
            flow.packet_count
            for flow in flows
        )

        min_packets = min(
            flow.packet_count
            for flow in flows
        )

        average_packets = (
            sum(
                flow.packet_count
                for flow in flows
            )
            / len(flows)
        )

        print("\n========== FLOW STATISTICS ==========\n")

        print(f"Minimum Packets In Flow : {min_packets}")
        print(f"Maximum Packets In Flow : {max_packets}")
        print(f"Average Packets In Flow : {average_packets:.2f}")

    else:

        print("No flows found.")

    print("\n========== PIPELINE SUMMARY ==========\n")

    print(f"Total Raw Packets      : {total_packets_seen}")
    print(f"Successfully Parsed    : {parsed_packets}")
    print(f"Successfully Processed : {processed_packets}")
    print(f"Total Flows Created    : {manager.flow_count()}")

    print("\n========== VALIDATION ==========\n")

    flow_packet_sum = sum(
        flow.packet_count
        for flow in flows
    )

    flow_byte_sum = sum(
        flow.total_bytes
        for flow in flows
    )

    print(f"Packet Counter         : {processed_packets}")
    print(f"Flow Packet Sum        : {flow_packet_sum}")
    print(f"Packet Count Match     : {processed_packets == flow_packet_sum}")

    print(f"Total Bytes In Flows   : {flow_byte_sum}")

    print("\n========== BYTE VALIDATION ==========\n")

    byte_validation_passed = True

    for flow in flows:

        calculated_bytes = sum(
            pkt["packet_len"]
            for pkt in flow.packets
        )

        if calculated_bytes != flow.total_bytes:

            byte_validation_passed = False

            print("BYTE COUNT ERROR")
            print("Flow Key:", flow.flow_key)
            print("Stored Bytes:", flow.total_bytes)
            print("Calculated Bytes:", calculated_bytes)

            break

    print(f"Byte Validation Passed : {byte_validation_passed}")

    print("\n========== TOP 10 FLOWS ==========\n")

    top_flows = sorted(
        flows,
        key=lambda flow: flow.packet_count,
        reverse=True
    )[:10]

    for i, flow in enumerate(top_flows, start=1):

        print(f"Flow #{i}")

        print(f"Key          : {flow.flow_key}")
        print(f"Packets      : {flow.packet_count}")
        print(f"Bytes        : {flow.total_bytes}")
        print(f"Start Time   : {flow.start_time}")
        print(f"Last Seen    : {flow.last_seen}")

        print()

    print("========== TEST COMPLETE ==========")


if __name__ == "__main__":
    main()