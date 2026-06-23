from scapy.all import rdpcap

from capture.packet_parser import PacketParser
from flow.flow_manager import FlowManager
from flow.sequence_builder import SequenceBuilder
from preprocessing.handshake_detector import HandshakeDetector
from preprocessing.normalizer import TemporalNormalizer


PCAP_FILE = (
    "datasets/raw_pcaps/iscx_official/NonVPN-PCAPs-02/ftps_up_2a.pcap"
)


def main():

    print("\n========== LOADING PCAP ==========")

    raw_packets = rdpcap(
        PCAP_FILE
    )

    print(
        "Raw packets:",
        len(raw_packets)
    )

    # ==================================================
    # Modules
    # ==================================================

    packet_parser = PacketParser()

    flow_manager = FlowManager()

    handshake_detector = (
        HandshakeDetector()
    )

    sequence_builder = (
        SequenceBuilder()
    )

    strict_normalizer = (
        TemporalNormalizer(
            mode="strict"
        )
    )

    fallback_normalizer = (
        TemporalNormalizer(
            mode="fallback"
        )
    )

    # ==================================================
    # Parse Packets
    # ==================================================

    parsed_count = 0

    for raw_packet in raw_packets:

        parsed_packet = (
            packet_parser.parse_packet(
                raw_packet
            )
        )

        if parsed_packet is None:
            continue

        parsed_count += 1

        flow_manager.process_packet(
            parsed_packet
        )

    print(
        "Parsed packets:",
        parsed_count
    )

    # ==================================================
    # Get Flows
    # ==================================================

    flows = (
        flow_manager.get_all_flows()
    )

    print(
        "Flows created:",
        len(flows)
    )

    # ==================================================
    # Handshake Detection
    # ==================================================

    tcp_count = 0
    tls_count = 0
    no_baseline_count = 0

    for flow in flows:

        handshake_detector.process_flow(
            flow
        )

        baseline_info = getattr(
            flow,
            "baseline_info",
            {}
        )

        if baseline_info.get(
            "baseline_available",
            False
        ):

            baseline_type = (
                baseline_info.get(
                    "baseline_type"
                )
            )

            if baseline_type == "tcp":
                tcp_count += 1

            elif baseline_type == "tls":
                tls_count += 1

        else:
            no_baseline_count += 1

    print("\n========== HANDSHAKES ==========")

    print(
        "TCP baselines:",
        tcp_count
    )

    print(
        "TLS baselines:",
        tls_count
    )

    print(
        "No baseline:",
        no_baseline_count
    )

    # ==================================================
    # Build Sequences
    # ==================================================

    sequence_records = []

    for flow in flows:

        sequence_record = (
            sequence_builder.build_sequence(
                flow
            )
        )

        if sequence_record:

            sequence_records.append(
                sequence_record
            )

    print(
        "\nSequences built:",
        len(sequence_records)
    )

    # ==================================================
    # Strict vs Fallback
    # ==================================================

    strict_normalized = 0
    fallback_normalized = 0

    for sequence_record in sequence_records:

        strict_result = (
            strict_normalizer.process_sequence(
                sequence_record
            )
        )

        fallback_result = (
            fallback_normalizer.process_sequence(
                sequence_record
            )
        )

        if strict_result[
            "normalization_applied"
        ]:
            strict_normalized += 1

        if fallback_result[
            "normalization_applied"
        ]:
            fallback_normalized += 1

    print(
        "\n========== STRICT vs FALLBACK =========="
    )

    print(
        "Strict Normalized:",
        strict_normalized
    )

    print(
        "Fallback Normalized:",
        fallback_normalized
    )

    # ==================================================
    # Detailed Normalization Analysis
    # ==================================================

    normalized_count = 0

    tcp_normalized = 0
    tls_normalized = 0
    proxy_normalized = 0
    none_normalized = 0

    baseline_values = []

    print(
        "\n========== NORMALIZATION =========="
    )

    for idx, sequence_record in enumerate(
        sequence_records,
        start=1
    ):

        result = (
            fallback_normalizer.process_sequence(
                sequence_record
            )
        )

        latency = result[
            "baseline_latency"
        ]

        if latency is not None:

            baseline_values.append(
                latency
            )

        if result[
            "normalization_applied"
        ]:
            normalized_count += 1

        baseline_type = (
            result[
                "baseline_type"
            ]
        )

        if baseline_type == "tcp":
            tcp_normalized += 1

        elif baseline_type == "tls":
            tls_normalized += 1

        elif baseline_type == "proxy":
            proxy_normalized += 1

        else:
            none_normalized += 1

        # Show only first 5 flows
        if idx <= 5:

            print(
                f"\nFlow #{idx}"
            )

            print(
                "Flow Key:",
                sequence_record.flow_key
            )

            print(
                "Baseline Type:",
                result[
                    "baseline_type"
                ]
            )

            print(
                "Baseline Latency:",
                result[
                    "baseline_latency"
                ]
            )

            print(
                "Normalization Applied:",
                result[
                    "normalization_applied"
                ]
            )

            raw_seq = result[
                "raw_sequence"
            ]

            norm_seq = result[
                "normalized_sequence"
            ]

            print(
                "\nRAW vs NORMALIZED"
            )

            for i in range(
                min(3, len(raw_seq))
            ):

                print(
                    f"\nPacket {i+1}"
                )

                print(
                    "Raw Relative:",
                    raw_seq[i][
                        "relative_timestamp"
                    ]
                )

                print(
                    "Normalized Relative:",
                    norm_seq[i][
                        "relative_timestamp"
                    ]
                )

                print(
                    "Raw IAT:",
                    raw_seq[i][
                        "inter_arrival_time"
                    ]
                )

                print(
                    "Normalized IAT:",
                    norm_seq[i][
                        "inter_arrival_time"
                    ]
                )

            # Formula Validation
            if latency is not None:

                print(
                    "\nBASELINE VALIDATION"
                )

                for i in range(
                    min(3, len(raw_seq))
                ):

                    raw_val = raw_seq[i][
                        "relative_timestamp"
                    ]

                    expected = (
                        raw_val / latency
                    )

                    actual = norm_seq[i][
                        "relative_timestamp"
                    ]

                    print(
                        f"Packet {i+1}: "
                        f"Expected={expected:.6f} "
                        f"Actual={actual:.6f}"
                    )

    # ==================================================
    # Final Summary
    # ==================================================

    print(
        "\n========== FINAL SUMMARY =========="
    )

    print(
        "Flows:",
        len(flows)
    )

    print(
        "Sequences:",
        len(sequence_records)
    )

    print(
        "Normalized:",
        normalized_count
    )

    print()

    print(
        "TCP normalized:",
        tcp_normalized
    )

    print(
        "TLS normalized:",
        tls_normalized
    )

    print(
        "Proxy normalized:",
        proxy_normalized
    )

    print(
        "No normalization:",
        none_normalized
    )

    # ==================================================
    # Baseline Statistics
    # ==================================================

    if baseline_values:

        print(
            "\n========== BASELINE STATS =========="
        )

        print(
            "Count:",
            len(baseline_values)
        )

        print(
            "Min:",
            min(baseline_values)
        )

        print(
            "Max:",
            max(baseline_values)
        )

        print(
            "Average:",
            sum(baseline_values)
            / len(baseline_values)
        )


if __name__ == "__main__":
    main()