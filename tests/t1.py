from capture.packet_parser import PacketParser
from flow.flow_manager import FlowManager
from flow.sequence_builder import SequenceBuilder
from preprocessing.handshake_detector import HandshakeDetector
import statistics
import csv

from scapy.all import PcapReader


PCAP_PATH = "datasets/raw_pcaps/iscx_official/NonVPN-PCAPs-02/hangouts_video2a.pcap"


parser = PacketParser()
flow_manager = FlowManager()
sequence_builder = SequenceBuilder()
handshake_detector = HandshakeDetector()


print("=" * 70)
print("NORMALIZATION ANALYSIS")
print("=" * 70)


with PcapReader(PCAP_PATH) as reader:

    for raw_packet in reader:

        packet = parser.parse_packet(raw_packet)

        if packet is not None:
            flow_manager.process_packet(packet)


flows = flow_manager.get_all_flows()

print("Flows :", len(flows))


sequences = sequence_builder.build_sequences(flows)

print("Sequences :", len(sequences))


sequences = handshake_detector.detect_handshakes(
    sequences
)

all_baselines = []
all_mean_iats = []
all_median_iats = []
all_max_iats = []
all_flow_durations = []

count = 0

for sequence in sequences:

    if count == 200:
        break

    print("\n" + "=" * 60)

    print("FLOW :", sequence.flow_key)

    baseline = sequence.tcp_baseline

    if baseline is None:
        baseline = sequence.tls_baseline

    print("Baseline :", baseline)

    packets = sequence.sequence

    iats = []
    timestamps = []

    for packet in packets:

        iats.append(packet["inter_arrival_time"])
        timestamps.append(packet["relative_timestamp"])

    mean_iat = statistics.mean(iats)
    median_iat = statistics.median(iats)
    max_iat = max(iats)
    flow_duration = timestamps[-1]

    print("Mean IAT :", mean_iat)
    print("Median IAT :", median_iat)
    print("Max IAT :", max_iat)
    print("Flow Duration :", flow_duration)

    if baseline is not None:

        print(
            "Mean IAT / Baseline :",
            mean_iat / baseline
        )

        all_baselines.append(baseline)
        all_mean_iats.append(mean_iat)
        all_median_iats.append(median_iat)
        all_max_iats.append(max_iat)
        all_flow_durations.append(flow_duration)

    count += 1
print("\n")
print("=" * 70)
print("SUMMARY")
print("=" * 70)

print("Flows Analysed :", len(all_baselines))

print("Average Baseline :", statistics.mean(all_baselines))
print("Average Mean IAT :", statistics.mean(all_mean_iats))
print("Average Flow Duration :", statistics.mean(all_flow_durations))

with open(
    "tests/normalization_analysis.csv",
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "baseline",
        "mean_iat",
        "median_iat",
        "max_iat",
        "flow_duration",
    ])

    for i in range(len(all_baselines)):

        writer.writerow([
            all_baselines[i],
            all_mean_iats[i],
            all_median_iats[i],
            all_max_iats[i],
            all_flow_durations[i],
        ])

print("\nCSV Saved Successfully.")