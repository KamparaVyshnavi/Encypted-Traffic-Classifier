from scapy.all import rdpcap

from capture.packet_parser import PacketParser
from flow.flow_manager import FlowManager
from preprocessing.handshake_detector import HandshakeDetector


PCAP_FILE = (
    "datasets/raw_pcaps/iscx_official/NonVPN-PCAPs-02/ftps_up_2a.pcap"
)


def main():

    packet_parser = PacketParser()

    flow_manager = FlowManager()

    handshake_detector = (
        HandshakeDetector()
    )

    print("Loading PCAP...")

    raw_packets = rdpcap(
        PCAP_FILE
    )

    for raw_packet in raw_packets:

        parsed_packet = (
            packet_parser.parse_packet(
                raw_packet
            )
        )

        if parsed_packet is None:
            continue

        flow_manager.process_packet(
            parsed_packet
        )

    flows = (
        flow_manager.get_all_flows()
    )

    latencies = []

    print(
        "\n========== BASELINE DETAILS =========="
    )

    for flow in flows:

        handshake_detector.process_flow(
            flow
        )

        baseline_info = getattr(
            flow,
            "baseline_info",
            {}
        )

        if not baseline_info.get(
            "baseline_available",
            False
        ):
            continue

        latency = baseline_info.get(
            "baseline_latency"
        )

        baseline_type = baseline_info.get(
            "baseline_type"
        )

        latencies.append(
            latency
        )

        print(
            f"{baseline_type:5s} "
            f"{latency:.9f}"
        )

    if latencies:

        print(
            "\n========== STATS =========="
        )

        print(
            "Count:",
            len(latencies)
        )

        print(
            "Min:",
            min(latencies)
        )

        print(
            "Max:",
            max(latencies)
        )

        print(
            "Average:",
            sum(latencies)
            / len(latencies)
        )

    else:

        print(
            "No baselines found."
        )


if __name__ == "__main__":
    main()