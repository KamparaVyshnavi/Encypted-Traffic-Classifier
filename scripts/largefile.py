import csv
import json
from pathlib import Path

import numpy as np
from scapy.utils import PcapReader

from capture.packet_parser import PacketParser
from flow.flow_manager import FlowManager
from flow.sequence_builder import SequenceBuilder

from preprocessing.handshake_detector import HandshakeDetector
from preprocessing.normalizer import TemporalNormalizer
from preprocessing.feature_encoder import FeatureEncoder


class DatasetGenerator:

    def __init__(
        self,
        raw_dataset_dir,
        output_dir,
        normalization_mode="fallback",
    ):

        self.raw_dataset_dir = Path(raw_dataset_dir)
        self.output_dir = Path(output_dir)

        self.sequence_dir = (
            self.output_dir /
            "sequences"
        )

        self.sequence_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.labels_file = (
            self.output_dir /
            "labels.csv"
        )

        self.metadata_file = (
            self.output_dir /
            "metadata.json"
        )

        # -----------------------------
        # Pipeline Modules
        # -----------------------------

        self.packet_parser = PacketParser()

        self.flow_manager = FlowManager()

        self.sequence_builder = SequenceBuilder()

        self.handshake_detector = HandshakeDetector()

        self.normalizer = TemporalNormalizer(
            mode=normalization_mode
        )

        self.feature_encoder = FeatureEncoder()

        # -----------------------------
        # Dataset State
        # -----------------------------

        self.sample_index = 0

        self.label_rows = []

        self.stats = {

            "pcaps_processed": 0,

            "packets_read": 0,

            "packets_parsed": 0,

            "flows_created": 0,

            "sequences_generated": 0,

            "normalized": 0,

            "strict": 0,

            "fallback": 0,

            "discarded": 0,

            "samples_saved": 0
        }

    # =====================================================

    def run(self):

        pcap_files = sorted(
            self.raw_dataset_dir.rglob("*.pcap")
        )

        print()

        print("=" * 60)
        print("DATASET GENERATION")
        print("=" * 60)

        print()

        print(
            f"Found {len(pcap_files)} PCAP files"
        )

        print()

        for pcap in pcap_files:

            print(
                f"Processing : {pcap.name}"
            )

            self.process_pcap(
                pcap
            )

            print()

    # =====================================================
    def get_label(self, filename):

        name = filename.lower()

        # Remove vpn_ prefix
        if name.startswith("vpn_"):
            name = name[4:]

        LABEL_MAP = {

            # =============================
            # Streaming
            # =============================
            "youtubehtml5": "Streaming",
            "facebook_video": "Streaming",
            "hangouts_video": "Streaming",
            "netflix": "Streaming",
            "spotify": "Streaming",
            "youtube": "Streaming",
            "vimeo": "Streaming",

            # =============================
            # VoIP
            # =============================
            "facebook_audio": "VoIP",
            "hangouts_audio": "VoIP",
            "skype_audio": "VoIP",
            "voipbuster": "VoIP",

            # =============================
            # File Transfer
            # =============================
            "skype_file": "FileTransfer",
            "browsing": "Browsing",
            "ftps": "FileTransfer",
            "sftp": "FileTransfer",
            "ftp": "FileTransfer",
            "scp": "FileTransfer",

            # =============================
            # Chat
            # =============================
            "facebook_chat": "Chat",
            "hangouts_chat": "Chat",
            "hangout_chat": "Chat",
            "gmailchat": "Chat",
            "skype_chat": "Chat",
            "facebookchat": "Chat",
            "aim_chat": "Chat",
            "icq_chat": "Chat",
            "aimchat": "Chat",
            "icqchat": "Chat",

            # =============================
            # Email
            # =============================
            "email": "Email",

            # =============================
            # P2P
            # =============================
            "bittorrent": "P2P",

            "browssing_firefox": "Browsing",
            "browsing_chrome": "Browsing",
            "browsing": "Browsing",
        }

        for prefix, label in LABEL_MAP.items():

            if name.startswith(prefix):
                return label

        return "Unknown"
    # ----------------------------------------------------
    def process_pcap(
        self,
        pcap_path
    ):

        self.stats[
            "pcaps_processed"
        ] += 1

        self.flow_manager.clear()

        label = self.get_label(pcap_path.stem)

        # ----------------------------------
        # Read packets
        # ----------------------------------

        with PcapReader(
            str(pcap_path)
        ) as reader:

            for raw_packet in reader:

                self.stats[
                    "packets_read"
                ] += 1

                if self.stats["packets_read"] % 100000 == 0:

                    print(
                    f"[{pcap_path.name}] "
                    f"Packets: {self.stats['packets_read']:,} | "
                    f"Active Flows: {self.flow_manager.flow_count()}"
                )


                parsed_packet = (
                    self.packet_parser
                    .parse_packet(
                        raw_packet
                    )
                )

                if parsed_packet is None:
                    continue

                self.stats[
                    "packets_parsed"
                ] += 1

                self.flow_manager.process_packet(
                    parsed_packet
                )

        # ----------------------------------
        # Build flows
        # ----------------------------------

        flows = (
            self.flow_manager
            .get_all_flows()
        )

        self.stats[
            "flows_created"
        ] += len(flows)

        # ----------------------------------
        # Detect handshakes
        # ----------------------------------

        for flow in flows:

            self.handshake_detector.process_flow(
                flow
            )

        # ----------------------------------
        # Build sequences
        # ----------------------------------

        sequences = (
            self.sequence_builder
            .build_sequences(
                flows
            )
        )

        self.stats[
            "sequences_generated"
        ] += len(sequences)

        # ----------------------------------
        # Normalize
        # ----------------------------------

        encoded_sequences = []

        for sequence in sequences:

            result = (
                self.normalizer
                .process_sequence(
                    sequence
                )
            )

            if result[
                "normalization_applied"
            ]:

                self.stats[
                    "normalized"
                ] += 1

                if (
                    result[
                        "baseline_type"
                    ]
                    == "proxy"
                ):

                    self.stats[
                        "fallback"
                    ] += 1

                else:

                    self.stats[
                        "strict"
                    ] += 1

            encoded = (
                self.feature_encoder
                .encode_sequence(
                    sequence
                )
            )

            encoded_sequences.append(
                encoded
            )

        # ----------------------------------
        # Save Dataset
        # ----------------------------------

        for encoded in encoded_sequences:

            feature_matrix = np.array(
                encoded.features,
                dtype=np.float32
            )

            sample_name = (
                f"sample_"
                f"{self.sample_index:07d}"
            )

            file_path = (
                self.sequence_dir /
                f"{sample_name}.npy"
            )

            np.save(
                file_path,
                feature_matrix
            )

            self.label_rows.append(

                (
                    sample_name,
                    label
                )

            )

            self.sample_index += 1

            self.stats[
                "samples_saved"
            ] += 1

        print(
            f"Packets Parsed      : "
            f"{self.stats['packets_parsed']}"
        )

        print(
            f"Flows Created       : "
            f"{len(flows)}"
        )

        print(
            f"Sequences Generated : "
            f"{len(sequences)}"
        )

        print(
            f"Samples Saved       : "
            f"{len(encoded_sequences)}"
        )
    
        # =====================================================
    # Save Labels
    # =====================================================

    def save_labels(self):

        with open(
            self.labels_file,
            "w",
            newline=""
        ) as csvfile:

            writer = csv.writer(csvfile)

            writer.writerow(
                [
                    "sample",
                    "label"
                ]
            )

            for row in self.label_rows:
                writer.writerow(row)

    # =====================================================
    # Save Metadata
    # =====================================================

    def save_metadata(self):

        class_distribution = {}

        for _, label in self.label_rows:

            class_distribution[label] = (
                class_distribution.get(
                    label,
                    0
                ) + 1
            )

        metadata = {

            "dataset_name":
                "Processed Encrypted Traffic Dataset",

            "total_pcaps":
                self.stats[
                    "pcaps_processed"
                ],

            "total_packets_read":
                self.stats[
                    "packets_read"
                ],

            "total_packets_parsed":
                self.stats[
                    "packets_parsed"
                ],

            "total_flows":
                self.stats[
                    "flows_created"
                ],

            "total_sequences":
                self.stats[
                    "sequences_generated"
                ],

            "samples_saved":
                self.stats[
                    "samples_saved"
                ],

            "normalization":{

                "total_normalized":
                    self.stats[
                        "normalized"
                    ],

                "strict":
                    self.stats[
                        "strict"
                    ],

                "fallback":
                    self.stats[
                        "fallback"
                    ]
            },

            "feature_dimension":
                self.feature_encoder.get_feature_dimension(),

            "sequence_length":
                self.sequence_builder.sequence_length,

            "classes":
                sorted(
                    class_distribution.keys()
                ),

            "class_distribution":
                class_distribution
        }

        with open(
            self.metadata_file,
            "w"
        ) as fp:

            json.dump(
                metadata,
                fp,
                indent=4
            )

    # =====================================================
    # Dataset Summary
    # =====================================================

    def print_summary(self):

        print()

        print("=" * 60)
        print("DATASET SUMMARY")
        print("=" * 60)

        print()

        print(
            f"PCAPs Processed      : "
            f"{self.stats['pcaps_processed']}"
        )

        print(
            f"Packets Read         : "
            f"{self.stats['packets_read']}"
        )

        print(
            f"Packets Parsed       : "
            f"{self.stats['packets_parsed']}"
        )

        print(
            f"Flows Created        : "
            f"{self.stats['flows_created']}"
        )

        print(
            f"Sequences Generated  : "
            f"{self.stats['sequences_generated']}"
        )

        print(
            f"Samples Saved        : "
            f"{self.stats['samples_saved']}"
        )

        print()

        print(
            f"Normalized Sequences : "
            f"{self.stats['normalized']}"
        )

        print(
            f"Strict Baselines     : "
            f"{self.stats['strict']}"
        )

        print(
            f"Fallback Baselines   : "
            f"{self.stats['fallback']}"
        )

        print()

        print(
            f"Output Directory     : "
            f"{self.output_dir}"
        )

        print()

    # =====================================================
    # Finalize
    # =====================================================

    def finalize(self):

        self.save_labels()

        self.save_metadata()

        self.print_summary()


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    generator = DatasetGenerator(

        raw_dataset_dir=(
            "datasets/raw_pcaps/iscx_kagglevpn326"
        ),

        output_dir=(
            "datasets/large_pcap_output"
        ),

        normalization_mode="fallback"
    )

    generator.run()

    generator.finalize()