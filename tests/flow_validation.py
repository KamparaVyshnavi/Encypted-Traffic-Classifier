# tests/test_flow_validation.py

from flow.flow_manager import FlowManager


def main():

    manager = FlowManager()

    packets = [

        {
            "timestamp": 1.0,
            "src_ip": "10.0.0.1",
            "dst_ip": "10.0.0.2",
            "src_port": 1111,
            "dst_port": 443,
            "protocol": "TCP",
            "packet_len": 100,
        },

        {
            "timestamp": 2.0,
            "src_ip": "10.0.0.1",
            "dst_ip": "10.0.0.2",
            "src_port": 1111,
            "dst_port": 443,
            "protocol": "TCP",
            "packet_len": 200,
        },

        {
            "timestamp": 3.0,
            "src_ip": "10.0.0.2",
            "dst_ip": "10.0.0.1",
            "src_port": 443,
            "dst_port": 1111,
            "protocol": "TCP",
            "packet_len": 150,
        },

        {
            "timestamp": 4.0,
            "src_ip": "10.0.0.1",
            "dst_ip": "10.0.0.2",
            "src_port": 2222,
            "dst_port": 443,
            "protocol": "TCP",
            "packet_len": 120,
        }
    ]

    total_packets = 0

    for packet in packets:
        result = manager.process_packet(packet)

        assert result["status"] == "processed"

        total_packets += 1

    print(f"Total Packets Processed: {total_packets}")
    print(f"Total Flows Created: {manager.flow_count()}")

    flows = manager.get_all_flows()

    for flow in flows:
        print()
        print("Flow Key:", flow.flow_key)
        print("Packet Count:", flow.packet_count)
        print("Total Bytes:", flow.total_bytes)

    packet_sum = sum(flow.packet_count for flow in flows)
    byte_sum = sum(flow.total_bytes for flow in flows)

    print()
    print("Packet Sum Across Flows:", packet_sum)
    print("Byte Sum Across Flows:", byte_sum)


if __name__ == "__main__":
    main()