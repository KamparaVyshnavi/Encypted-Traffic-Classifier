from scapy.utils import PcapReader

from capture.packet_parser import PacketParser

from flow.flow_manager import FlowManager
from flow.sequence_builder import SequenceBuilder

from preprocessing.handshake_detector import HandshakeDetector
from preprocessing.normalizer import TemporalNormalizer
from preprocessing.feature_encoder import FeatureEncoder
from preprocessing.tensor_converter import TensorConverter


def main():

    pcap_path = r"datasets/raw_pcaps/iscx_official/NonVPN-PCAPs-03/scpDown3.pcap"

    print("=" * 80)
    print("Loading PCAP...")
    print("=" * 80)

    parser = PacketParser()
    flow_manager = FlowManager()

    packet_count = 0

    # ----------------------------------------------------------
    # Parse PCAP and Build Flows
    # ----------------------------------------------------------

    with PcapReader(pcap_path) as pcap:

        for raw_packet in pcap:

            parsed_packet = parser.parse_packet(
                raw_packet
            )

            if parsed_packet is None:
                continue

            flow_manager.process_packet(
                parsed_packet
            )

            packet_count += 1

    print(f"Packets Parsed : {packet_count}")

    flows = flow_manager.get_all_flows()

    print(f"Flows Built    : {len(flows)}")

    # ----------------------------------------------------------
    # Initialize Pipeline Modules
    # ----------------------------------------------------------

    handshake_detector = HandshakeDetector()

    sequence_builder = SequenceBuilder()

    normalizer = TemporalNormalizer()

    encoder = FeatureEncoder()

    converter = TensorConverter()

    print()
    print("=" * 80)
    print("Running Complete Preprocessing Pipeline")
    print("=" * 80)

    processed_sequences = 0

    # ----------------------------------------------------------
    # Process Every Flow
    # ----------------------------------------------------------

    for flow_index, flow in enumerate(flows, start=1):

        print()
        print("-" * 80)
        print(f"Flow {flow_index}")
        print("-" * 80)

        # ----------------------------------------------
        # Handshake Detection
        # ----------------------------------------------

        handshake_detector.process_flow(flow)

        # ----------------------------------------------
        # Sequence Construction
        # ----------------------------------------------

        sequence = sequence_builder.build_sequence(
            flow
        )

        if sequence is None:
            print("Skipped (sequence not generated)")
            continue

        # ----------------------------------------------
        # Temporal Normalization
        # ----------------------------------------------

        normalization_result = (
            normalizer.process_sequence(
                sequence
            )
        )

        # ----------------------------------------------
        # Feature Encoding
        # ----------------------------------------------

        encoded_sequence = (
            encoder.encode_sequence(
                sequence
            )
        )

        # ----------------------------------------------
        # Tensor Conversion
        # ----------------------------------------------

        tensor_sequence = (
            converter.convert_sequence(
                encoded_sequence
            )
        )

        processed_sequences += 1

        # ----------------------------------------------
        # Results
        # ----------------------------------------------

        print("Flow Key              :", tensor_sequence.flow_key)

        print("Sequence Length       :", tensor_sequence.sequence_length)

        print("Feature Dimension     :", tensor_sequence.feature_dimension)

        print("Valid Exit Points     :", tensor_sequence.valid_exit_points)

        print("Closed Flow           :", tensor_sequence.is_closed)

        print("Padding Applied       :", tensor_sequence.is_padded)

        print()

        print("Baseline Available    :",
              normalization_result["baseline_available"])

        print("Baseline Type         :",
              normalization_result["baseline_type"])

        print("Baseline Latency      :",
              normalization_result["baseline_latency"])

        print("Normalization Applied :",
              normalization_result["normalization_applied"])

        print()

        print("Tensor Shape          :",
              tuple(tensor_sequence.tensor.shape))

        print("Tensor Dtype          :",
              tensor_sequence.tensor.dtype)

        print()

        print("Tensor")

        print(tensor_sequence.tensor)

    print()
    print("=" * 80)
    print("Pipeline Summary")
    print("=" * 80)

    print("Packets Parsed        :", packet_count)

    print("Flows Built           :", len(flows))

    print("Sequences Processed   :", processed_sequences)

    print()
    print("Pipeline Executed Successfully")


if __name__ == "__main__":
    main()