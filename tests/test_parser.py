from scapy.all import PcapReader
from capture.packet_parser import PacketParser

parser = PacketParser()

pcap_path= "datasets/raw_pcaps/USTC_TFC/Benign/FTP.pcap"
count=0

for packet in PcapReader(pcap_path):

    parsed = parser.parse_packet(packet)

    if parsed:
        print(parsed)
        count+=1
        if(count==3):
            break