from scapy.all import rdpcap

from capture.packet_parser import PacketParser
from flow.flow_manager import FlowManager
from flow.sequence_builder import (
    SequenceBuilder,
    SequenceBuilderConfig,
)
from preprocessing.feature_encoder import (
    FeatureEncoder,
)


PCAP_FILE = r"datasets/raw_pcaps/iscx_official/VPN-PCAPS-01/vpn_ftps_A.pcap"


def main():

    print("=" * 60)
    print("FEATURE ENCODER INTEGRATION TEST")
    print("=" * 60)

    parser = PacketParser()

    sequence_builder = SequenceBuilder(
        SequenceBuilderConfig(
            sequence_length=20,
            exit_points=[5, 10, 15, 20],
        )
    )

    flow_manager = FlowManager(
        sequence_builder=sequence_builder
    )

    feature_encoder = FeatureEncoder()

    packets = rdpcap(PCAP_FILE)

    total_packets = 0
    parsed_packets = 0
    encoded_sequences = []

    for raw_packet in packets:

        total_packets += 1

        parsed_packet = parser.parse_packet(
            raw_packet
        )

        if parsed_packet is None:
            continue

        parsed_packets += 1

        result = flow_manager.process_packet(
            parsed_packet
        )

        sequences = result["sequences"]

        for sequence_record in sequences:

            encoded = (
                feature_encoder.encode_sequence(
                    sequence_record
                )
            )

            encoded_sequences.append(
                encoded
            )

    print()
    print(f"Total packets: {total_packets}")
    print(f"Parsed packets: {parsed_packets}")
    print(
        f"Encoded sequences: "
        f"{len(encoded_sequences)}"
    )

    if not encoded_sequences:
        print()
        print(
            "No sequences generated."
        )
        print(
            "Try lowering exit_points "
            "or use a larger PCAP."
        )
        return

    print()
    print("=" * 60)
    print("FIRST ENCODED SEQUENCE")
    print("=" * 60)

    sequence = encoded_sequences[0]

    print(
        f"Flow Key: {sequence.flow_key}"
    )

    print(
        f"Sequence Length: "
        f"{sequence.sequence_length}"
    )

    print(
        f"Feature Dimension: "
        f"{sequence.feature_dimension}"
    )

    print(
        f"Exit Points: "
        f"{sequence.valid_exit_points}"
    )

    print(
        f"Is Padded: "
        f"{sequence.is_padded}"
    )

    print(
        f"Is Closed: "
        f"{sequence.is_closed}"
    )

    print()
    print(
        "First 10 encoded packets:"
    )
    print()

    for i, feature_vector in enumerate(
        sequence.features[:10]
    ):
        print(
            f"{i + 1:02d}: "
            f"{feature_vector}"
        )

    print()
    print("=" * 60)
    print("FEATURE ORDER")
    print("=" * 60)

    print(
        "[packet_len, direction, "
        "protocol_id, tcp_flags, timestamp]"
    )

    print()
    print("Integration test completed.")
    
    # ============================================================
    # BASIC SUMMARY
    # ============================================================

    print()
    print("=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)

    print(f"Total packets: {total_packets}")
    print(f"Parsed packets: {parsed_packets}")
    print(f"Encoded sequences: {len(encoded_sequences)}")

    # ============================================================
    # TEST 1 : FLOW STATISTICS
    # ============================================================

    print()
    print("=" * 60)
    print("TEST 1 - FLOW STATISTICS")
    print("=" * 60)

    all_flows = flow_manager.get_all_flows()

    print(f"Active Flows: {len(all_flows)}")

    flows_ge_5 = 0
    flows_ge_10 = 0
    flows_ge_15 = 0
    flows_ge_20 = 0

    for flow in all_flows:

        if flow.packet_count >= 5:
            flows_ge_5 += 1

        if flow.packet_count >= 10:
            flows_ge_10 += 1

        if flow.packet_count >= 15:
            flows_ge_15 += 1

        if flow.packet_count >= 20:
            flows_ge_20 += 1

    print(f"Flows >= 5 packets : {flows_ge_5}")
    print(f"Flows >= 10 packets: {flows_ge_10}")
    print(f"Flows >= 15 packets: {flows_ge_15}")
    print(f"Flows >= 20 packets: {flows_ge_20}")

    # ============================================================
    # TEST 2 : FLOW SIZE DISTRIBUTION
    # ============================================================

    print()
    print("=" * 60)
    print("TEST 2 - FLOW SIZE DISTRIBUTION")
    print("=" * 60)

    buckets = {
        "1": 0,
        "2-4": 0,
        "5-9": 0,
        "10-19": 0,
        "20+": 0,
    }

    for flow in all_flows:

        count = flow.packet_count

        if count == 1:
            buckets["1"] += 1

        elif count <= 4:
            buckets["2-4"] += 1

        elif count <= 9:
            buckets["5-9"] += 1

        elif count <= 19:
            buckets["10-19"] += 1

        else:
            buckets["20+"] += 1

    for bucket, value in buckets.items():
        print(f"{bucket:>5} : {value}")

    # ============================================================
    # TEST 3 : EXIT POINT DISTRIBUTION
    # ============================================================

    print()
    print("=" * 60)
    print("TEST 3 - EXIT POINT DISTRIBUTION")
    print("=" * 60)

    exit_stats = {}

    for seq in encoded_sequences:

        key = str(seq.valid_exit_points)

        exit_stats[key] = (
            exit_stats.get(key, 0) + 1
        )

    for key in sorted(exit_stats.keys()):
        print(f"{key:<20} : {exit_stats[key]}")

    # ============================================================
    # TEST 4 : TOP 10 LARGEST FLOWS
    # ============================================================

    print()
    print("=" * 60)
    print("TEST 4 - TOP 10 LARGEST FLOWS")
    print("=" * 60)

    largest_flows = sorted(
        all_flows,
        key=lambda flow: flow.packet_count,
        reverse=True,
    )

    for i, flow in enumerate(
        largest_flows[:10],
        start=1,
    ):

        print(
            f"{i:02d}. "
            f"Packets={flow.packet_count:<8} "
            f"Bytes={flow.total_bytes:<12} "
            f"Key={flow.flow_key}"
        )

    # ============================================================
    # TEST 5 : SAMPLE SEQUENCE PROGRESSION
    # ============================================================

    print()
    print("=" * 60)
    print("TEST 5 - SAMPLE SEQUENCE PROGRESSION")
    print("=" * 60)

    sample_count = min(
        20,
        len(encoded_sequences),
    )

    for i in range(sample_count):

        seq = encoded_sequences[i]

        print(
            f"{i+1:02d}. "
            f"ExitPoints={seq.valid_exit_points} "
            f"Padded={seq.is_padded} "
            f"Closed={seq.is_closed}"
        )

    # ============================================================
    # FEATURE SHAPE VALIDATION
    # ============================================================

    print()
    print("=" * 60)
    print("FEATURE SHAPE VALIDATION")
    print("=" * 60)

    shape_errors = 0

    for seq in encoded_sequences:

        for feature_vector in seq.features:

            if (
                len(feature_vector)
                != seq.feature_dimension
            ):
                shape_errors += 1

    print(
        f"Shape Errors: {shape_errors}"
    )

    if shape_errors == 0:
        print("PASS")
    else:
        print("FAIL")

    # ============================================================
    # FIRST ENCODED SEQUENCE
    # ============================================================

    if encoded_sequences:

        print()
        print("=" * 60)
        print("FIRST ENCODED SEQUENCE")
        print("=" * 60)

        sequence = encoded_sequences[0]

        print(
            f"Flow Key: {sequence.flow_key}"
        )

        print(
            f"Sequence Length: "
            f"{sequence.sequence_length}"
        )

        print(
            f"Feature Dimension: "
            f"{sequence.feature_dimension}"
        )

        print(
            f"Exit Points: "
            f"{sequence.valid_exit_points}"
        )

        print(
            f"Is Padded: "
            f"{sequence.is_padded}"
        )

        print(
            f"Is Closed: "
            f"{sequence.is_closed}"
        )

        print()
        print(
            "First 10 encoded packets:"
        )
        print()

        for i, feature_vector in enumerate(
            sequence.features[:10]
        ):
            print(
                f"{i+1:02d}: "
                f"{feature_vector}"
            )

        print()
        print(
            "[packet_len, direction, "
            "protocol_id, tcp_flags, timestamp]"
        )

    print()
    print("=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    main()