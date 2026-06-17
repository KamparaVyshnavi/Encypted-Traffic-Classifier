from scapy.all import PcapReader

from capture.packet_parser import PacketParser
from flow.flow_manager import FlowManager


PCAP_FILE = (
    "datasets/raw_pcaps/iscx_kagglevpn326/vpn_hangouts_chat1b.pcap"
)


def main():

    parser = PacketParser()
    manager = FlowManager()

    processed_packets = 0

    with PcapReader(PCAP_FILE) as reader:

        for packet in reader:

            parsed_packet = parser.parse_packet(packet)

            if parsed_packet is None:
                continue

            result = manager.process_packet(parsed_packet)

            if result["status"] != "processed":
                continue

            processed_packets += 1

    flows = manager.get_all_flows()

    thresholds = [
        5,
        10,
        20,
        30,
        50,
        100,
        200,
        500,
        1000,
    ]

    print("\n========== DATASET SUMMARY ==========\n")

    print(f"Processed Packets : {processed_packets}")
    print(f"Total Flows       : {len(flows)}")

    print("\n========== FLOW THRESHOLD ANALYSIS ==========\n")

    for threshold in thresholds:

        count = sum(
            1
            for flow in flows
            if flow.packet_count >= threshold
        )

        percentage = (
            (count / len(flows)) * 100
            if flows
            else 0
        )

        print(
            f"Flows >= {threshold:>4} packets : "
            f"{count:>6} "
            f"({percentage:6.2f}%)"
        )

    print("\n========== USABLE FOR N-PACKET MODEL ==========\n")

    n_values = [5, 10, 20, 30, 50]

    for n in n_values:

        usable = sum(
            1
            for flow in flows
            if flow.packet_count >= n
        )

        print(
            f"N = {n:>2}  -> "
            f"{usable:>6} usable flows"
        )

    print("\n========== TOP 20 LONGEST FLOWS ==========\n")

    top_flows = sorted(
        flows,
        key=lambda flow: flow.packet_count,
        reverse=True
    )[:20]

    for index, flow in enumerate(
        top_flows,
        start=1
    ):

        print(
            f"{index:>2}. "
            f"Packets={flow.packet_count:<8} "
            f"Key={flow.flow_key}"
        )


if __name__ == "__main__":
    main()