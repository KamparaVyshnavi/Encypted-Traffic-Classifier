"""
===============================================================================
Encrypted Traffic Classifier
===============================================================================

Entry point of the complete real-time encrypted traffic
classification system.

Pipeline
--------

Interface Manager
        ↓
Packet Sniffer
        ↓
Packet Parser
        ↓
Flow Manager
        ↓
Sequence Builder
        ↓
Handshake Detector
        ↓
Temporal Normalizer
        ↓
Feature Encoder
        ↓
Multi-Exit CNN
        ↓
Traffic Statistics

Version
-------

Version 1
    Console Interface

Future
------

Version 2
    Dashboard Interface
"""
import time
import torch
from pathlib import Path
from collections import defaultdict

from capture.interface_manager import InterfaceManager
from capture.packet_sniffer import PacketSniffer
from capture.packet_parser import PacketParser

from flow.flow_manager import FlowManager
from flow.sequence_builder import SequenceBuilder

from preprocessing.handshake_detector import HandshakeDetector
from preprocessing.normalizer import TemporalNormalizer
from preprocessing.feature_encoder import FeatureEncoder

from model.inference import MultiExitInference

from utils.config import (
    MULTI_EXIT_CHECKPOINT_DIR,
    MULTI_EXIT_BEST_MODEL_NAME,
)

# =============================================================================
# Main System
# =============================================================================


class EncryptedTrafficClassifier:

    """
    Complete real-time encrypted traffic classifier.
    """

    def __init__(self):

        # ------------------------------------------------------------
        # Statistics
        # ------------------------------------------------------------

        self.packet_count = 0

        self.completed_flows = 0

        self.active_flows = 0

        self.class_counter = defaultdict(int)

        self.exit_counter = defaultdict(int)

        self.total_inference_time = 0.0

        self.total_packets_used = 0

        # ------------------------------------------------------------
        # Core Modules
        # ------------------------------------------------------------

        self.interface_manager = InterfaceManager()

        self.packet_parser = PacketParser()

        self.flow_manager = FlowManager()

        self.sequence_builder = SequenceBuilder()

        self.handshake_detector = HandshakeDetector()

        self.normalizer = TemporalNormalizer()

        self.feature_encoder = FeatureEncoder()
 
        self.last_dashboard_update = time.time()
        self.label_map = {
            0: "Chat",
            1: "Email",
            2: "FileTransfer",
            3: "P2P",
            4: "Streaming",
            5: "VoIP",
        }

        checkpoint = (

            Path(MULTI_EXIT_CHECKPOINT_DIR)

            / MULTI_EXIT_BEST_MODEL_NAME

        )

        self.inference_engine = MultiExitInference(

            checkpoint_path=checkpoint,

        )

        self.sniffer = None

    # ------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------

    def initialize(self):

        print()

        print("=" * 70)
        print("ENCRYPTED TRAFFIC CLASSIFIER")
        print("=" * 70)

        print()

        print("Initializing modules...")

        print("✓ Interface Manager")

        print("✓ Packet Parser")

        print("✓ Flow Manager")

        print("✓ Sequence Builder")

        print("✓ Handshake Detector")

        print("✓ Temporal Normalizer")

        print("✓ Feature Encoder")

        print("✓ Multi-Exit CNN")

        print()

        print("System Ready.")

        print()

    # ------------------------------------------------------------
    # Interface Selection
    # ------------------------------------------------------------

    def select_interface(
    self,
    interface=None,
):

        if interface is None:

            interface = self.interface_manager.select_interface()

        self.sniffer = PacketSniffer(

            interface=interface,

            packet_callback=self.process_packet,

            filter_rule="tcp or udp",

        )

        print()

        print(f"Using Interface : {interface}")

        print()

    def start(
    self,
    interface,
):

        self.initialize()

        self.select_interface(interface)

        print("Starting live capture...")

        print()

        self.sniffer.start_capture()
    
    def stop(self):

        if self.sniffer is not None:

            self.sniffer.stop_capture()

    # ------------------------------------------------------------
    # Packet Processing
    # ------------------------------------------------------------
    def process_packet(
        self,
        packet,
    ):
        """
        Complete real-time pipeline.
        """

        # ------------------------------------------------------------
        # Packet Counter
        # ------------------------------------------------------------

        self.packet_count += 1

        # ------------------------------------------------------------
        # Parse Packet
        # ------------------------------------------------------------

        parsed_packet = self.packet_parser.parse_packet(packet)

        if parsed_packet is None:
            return

        # ------------------------------------------------------------
        # Flow Manager
        # ------------------------------------------------------------

        flow_result = self.flow_manager.process_packet(
            parsed_packet
        )

        if flow_result["status"] != "processed":
            return

        flow = flow_result["flow"]

        self.active_flows = self.flow_manager.flow_count()

        # ------------------------------------------------------------
        # Sequence Generation
        # ------------------------------------------------------------

        sequence_records = self.sequence_builder.build_ready_sequences(
            flow
        )

        if not sequence_records:
            return

        # ------------------------------------------------------------
        # Process every newly generated sequence
        # ------------------------------------------------------------

        for sequence_record in sequence_records:

            self.completed_flows += 1

            # --------------------------------------------------------
            # Handshake Analysis
            # --------------------------------------------------------

            self.handshake_detector.process_flow(flow)

            # baseline information becomes attached to flow

            sequence_record.baseline_info = getattr(
                flow,
                "baseline_info",
                {
                    "baseline_available": False
                }
            )

            # --------------------------------------------------------
            # Novelty-1
            # --------------------------------------------------------

            self.normalizer.process_sequence(
                sequence_record
            )

            # --------------------------------------------------------
            # Feature Encoding + Tensor Conversion
            # --------------------------------------------------------

            features = self.feature_encoder.encode_tensor(
                sequence_record
            )

            # --------------------------------------------------------
            # Multi-Exit Inference
            # --------------------------------------------------------

            start = time.perf_counter()

            result = self.inference_engine.predict(
                features,
                threshold=0.70,
            )


            end = time.perf_counter()

            inference_time = end - start
            
            prediction = self.label_map[
            result["prediction"]
            ]

            confidence = result["confidence"]

            exit_used = result["exit"]

            # --------------------------------------------------------
            # Statistics
            # --------------------------------------------------------

            self.total_inference_time += inference_time

            self.total_packets_used += {

                "exit1": 5,

                "exit2": 10,

                "exit3": 15,

                "final": 20,

            }[result["exit"]]

            self.exit_counter[
                result["exit"]
            ] += 1

            

            predicted_label = self.label_map[
                result["prediction"]
            ]

            self.class_counter[
                predicted_label
            ] += 1

            # --------------------------------------------------------
            # Save latest prediction
            # --------------------------------------------------------

            self.latest_prediction = {

                "prediction": predicted_label,

                "confidence": result["confidence"],

                "exit": result["exit"],

                "inference_time": inference_time,
            }
            if self.completed_flows:

                average_packets = (
                    self.total_packets_used
                    / self.completed_flows
                )

                latency_saved = (
                    (20 - average_packets)
                    / 20
                    * 100
                )

            else:

                average_packets = 20

                latency_saved = 0


            # ----------------------------------------
            # Refresh Console Dashboard Once Every Second
            # ----------------------------------------

            if time.time() - self.last_dashboard_update >= 1:

                self.print_dashboard()

                self.last_dashboard_update = time.time()
    # ------------------------------------------------------------
    # Run
    # ------------------------------------------------------------

    def run(self):

        self.start(None)

    
    # ------------------------------------------------------------
    # Traffic Distribution
    # ------------------------------------------------------------

    def print_dashboard(self):

        stats = self.get_statistics()

        import os

        os.system("cls" if os.name == "nt" else "clear")

        print("=" * 80)
        print("ENCRYPTED TRAFFIC CLASSIFIER")
        print("=" * 80)
        print()

        print("SYSTEM STATUS")
        print("-" * 80)

        print(f"Packets Captured     : {stats['packet_count']}")
        print(f"Active Flows         : {stats['active_flows']}")
        print(f"Completed Flows      : {stats['completed_flows']}")

        print()

        # --------------------------------------------------------
        # Traffic Distribution
        # --------------------------------------------------------

        print("FLOW CLASSIFICATION SUMMARY")
        print("-" * 80)

        total = sum(stats['class_counter'].values())

        classes = [

            "Streaming",

            "Chat",

            "VoIP",

            "FileTransfer",

            "Email",

            "P2P",

        ]

        for traffic_class in classes:

            count = stats['class_counter'].get(
                traffic_class,
                0,
            )

            percentage = (

                100 * count / total

                if total > 0

                else 0
            )

            print(
                f"{traffic_class:<15}"
                f"{count:>6}"
                f"   "
                f"{percentage:6.2f}%"
            )

        print()

        # --------------------------------------------------------
        # Latest Prediction
        # --------------------------------------------------------

        print("LATEST CLASSIFICATION")
        print("-" * 80)

        if hasattr(self, "latest_prediction"):

            latest = stats["latest_prediction"]

            print(
                f"Prediction      : "
                f"{latest['prediction']}"
            )

            print(
                f"Confidence      : "
                f"{latest['confidence']:.2%}"
            )

            print(
                f"Exit Used       : "
                f"{latest['exit']}"
            )

            print(
                f"Inference Time  : "
                f"{latest['inference_time']*1000:.3f} ms"
            )

        else:

            print("Waiting for first prediction...")

        print()

        # --------------------------------------------------------
        # Exit Usage
        # --------------------------------------------------------

        print("EARLY EXIT USAGE")
        print("-" * 80)

        total_exit = sum(
            stats["exit_counter"].values()
        )

        for exit_name in [

            "exit1",

            "exit2",

            "exit3",

            "final",

        ]:

            count = stats['exit_counter'].get(
                exit_name,
                0,
            )

            percentage = (

                100 * count / total_exit

                if total_exit

                else 0
            )

            print(
                f"{exit_name:<10}"
                f"{count:>6}"
                f"   "
                f"{percentage:6.2f}%"
            )

        print()

       # --------------------------------------------------------
        # Performance
        # --------------------------------------------------------

        print("PERFORMANCE")
        print("-" * 80)

        if stats["completed_flows"]:

            print(
                f"Average Inference : "
                f"{stats['average_latency']*1000:.3f} ms"
            )

            print(
                f"Average Packets   : "
                f"{stats['average_packets']:.2f} / 20"
            )

            print(
                f"Computation Saved : "
                f"{stats['latency_saved']:.2f}%"
            )

        print()

        print("=" * 80)
    
    # ------------------------------------------------------------
    # Runtime Statistics
    # ------------------------------------------------------------

    def get_statistics(self):

        if self.completed_flows:

            average_latency = (

                self.total_inference_time

                / self.completed_flows

            )

            average_packets = (

                self.total_packets_used

                / self.completed_flows

            )

            latency_saved = (

                (20 - average_packets)

                / 20

                * 100

            )

        else:

            average_latency = 0

            average_packets = 20

            latency_saved = 0

        return {

            "packet_count": self.packet_count,

            "active_flows": self.active_flows,

            "completed_flows": self.completed_flows,

            "class_counter": dict(self.class_counter),

            "exit_counter": dict(self.exit_counter),

            "latest_prediction": getattr(

                self,

                "latest_prediction",

                None,

            ),

            "average_latency": average_latency,

            "average_packets": average_packets,

            "latency_saved": latency_saved,
        }

# =============================================================================
# Entry Point
# =============================================================================

def main():

    classifier = EncryptedTrafficClassifier()

    classifier.run()


if __name__ == "__main__":

    main()