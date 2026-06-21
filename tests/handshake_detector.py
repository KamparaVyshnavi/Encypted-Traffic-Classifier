from scapy.all import rdpcap

from capture.packet_parser import PacketParser
from flow.flow_manager import FlowManager
from preprocessing.handshake_detector import HandshakeDetector


pcap_file = (
    r"datasets/raw_pcaps/iscx_official/VPN-PCAPs-01/vpn_hangouts_audio2.pcap"
)


def main():

    parser = PacketParser()
    flow_manager = FlowManager()
    detector = HandshakeDetector()

    print("=" * 80)
    print("READING PCAP")
    print("=" * 80)

    raw_packets = rdpcap(pcap_file)

    print(f"Raw Packets: {len(raw_packets)}")

    parsed_packets = 0

    for raw_packet in raw_packets:

        packet = parser.parse_packet(raw_packet)

        if packet is None:
            continue

        parsed_packets += 1

        flow_manager.process_packet(packet)

    flows = flow_manager.get_all_flows()

    print(f"Parsed Packets : {parsed_packets}")
    print(f"Total Flows    : {len(flows)}")

    if not flows:
        print("No flows created.")
        return

    print("\nSample Packet Structure:")
    print(type(flows[0].packets[0]))
    print(flows[0].packets[0])

    total_flows = len(flows)

    # ==========================================================
    # TCP Control Packet Statistics
    # ==========================================================

    syn_packets = 0
    syn_ack_packets = 0
    ack_packets = 0

    flows_with_syn = 0
    flows_with_syn_ack = 0
    flows_with_ack = 0

    candidate_tcp_handshake_flows = 0

    # ==========================================================
    # Handshake Detector Statistics
    # ==========================================================

    flows_with_handshake = 0
    flows_with_tcp = 0
    flows_with_tls = 0

    print()
    print("=" * 80)
    print("ANALYZING FLOWS")
    print("=" * 80)

    for idx, flow in enumerate(flows):

        if idx % 100 == 0:
            print(
                f"Processed "
                f"{idx}/{total_flows} flows..."
            )

        has_syn = False
        has_syn_ack = False
        has_ack = False

        # ------------------------------------------------------
        # Raw TCP Flag Analysis
        # ------------------------------------------------------

        for packet in flow.packets:

            if packet["protocol"] != "TCP":
                continue

            flags = packet["tcp_flags"]

            if flags == 2:
                syn_packets += 1
                has_syn = True

            elif flags == 18:
                syn_ack_packets += 1
                has_syn_ack = True

            elif flags == 16:
                ack_packets += 1
                has_ack = True

        if has_syn:
            flows_with_syn += 1

        if has_syn_ack:
            flows_with_syn_ack += 1

        if has_ack:
            flows_with_ack += 1

        if has_syn and has_syn_ack and has_ack:
            candidate_tcp_handshake_flows += 1

        # ------------------------------------------------------
        # Handshake Detector
        # ------------------------------------------------------

        try:

            result = detector.process_flow(flow)

            if result["handshake_detected"]:

                print(result["handshake_features"])

            if result["handshake_detected"]:
                flows_with_handshake += 1

            if len(result["tcp_handshake_packets"]) > 0:
                flows_with_tcp += 1

            if len(result["tls_handshake_packets"]) > 0:
                flows_with_tls += 1

        except Exception as e:

            print()
            print(
                f"Detector Error "
                f"on flow {idx}: {e}"
            )

            print(
                f"Flow packet count: "
                f"{len(flow.packets)}"
            )

            break

    # ==========================================================
    # TCP Statistics
    # ==========================================================

    print()
    print("=" * 80)
    print("TCP CONTROL PACKET ANALYSIS")
    print("=" * 80)

    print(f"Total SYN Packets             : {syn_packets}")
    print(f"Total SYN-ACK Packets         : {syn_ack_packets}")
    print(f"Total ACK Packets             : {ack_packets}")

    print()

    print(f"Flows With SYN                : {flows_with_syn}")
    print(f"Flows With SYN-ACK            : {flows_with_syn_ack}")
    print(f"Flows With ACK                : {flows_with_ack}")

    print()

    print(
        f"Flows With SYN+SYNACK+ACK     : "
        f"{candidate_tcp_handshake_flows}"
    )

    # ==========================================================
    # Detector Statistics
    # ==========================================================

    print()
    print("=" * 80)
    print("HANDSHAKE DETECTOR ANALYSIS")
    print("=" * 80)

    print(f"Total Flows                  : {total_flows}")
    print(f"Flows With Handshake         : {flows_with_handshake}")
    print(f"Flows With TCP Handshake     : {flows_with_tcp}")
    print(f"Flows With TLS Handshake     : {flows_with_tls}")

    if total_flows > 0:

        print(
            f"Handshake Coverage           : "
            f"{100 * flows_with_handshake / total_flows:.2f}%"
        )

        print(
            f"TCP Handshake Coverage       : "
            f"{100 * flows_with_tcp / total_flows:.2f}%"
        )

        print(
            f"TLS Handshake Coverage       : "
            f"{100 * flows_with_tls / total_flows:.2f}%"
        )


if __name__ == "__main__":
    main()