from scapy.all import sniff
from typing import Callable, Optional


class PacketSniffer:
    """
    Handles live packet capture from a selected network interface.

    Responsibilities:
    - start packet capture
    - stop packet capture
    - forward captured packets to callback function
    """

    def __init__(
        self,
        interface: str,
        packet_callback: Callable,
        packet_count: Optional[int] = None,
        timeout: Optional[int] = None,
        filter_rule: Optional[str] = None
    ):
        """
        Parameters:
        -----------
        interface : str
            Network interface to capture packets from.

        packet_callback : Callable
            Function that will process each captured packet.

        packet_count : Optional[int]
            Maximum number of packets to capture.
            None means unlimited capture.

        timeout : Optional[int]
            Capture duration in seconds.
            None means no timeout.

        filter_rule : Optional[str]
            BPF filter for packet filtering.
            Example:
                "tcp"
                "udp"
                "port 443"
        """

        self.interface = interface
        self.packet_callback = packet_callback
        self.packet_count = packet_count
        self.timeout = timeout
        self.filter_rule = filter_rule

        self.is_running = False

    def start_capture(self) -> None:
        """
        Starts live packet capture.
        """

        print(f"\nStarting packet capture on interface: {self.interface}")

        self.is_running = True

        sniff_arguments = {
        "iface": self.interface,
        "prn": self.handle_packet,
        "store": False
        }

        if self.packet_count is not None:
            sniff_arguments["count"] = self.packet_count

        if self.timeout is not None:
            sniff_arguments["timeout"] = self.timeout

        if self.filter_rule is not None:
            sniff_arguments["filter"] = self.filter_rule

        sniff(**sniff_arguments)

        self.is_running = False

        print("\nPacket capture stopped.")

    def handle_packet(self, packet) -> None:
        """
        Handles every captured packet.

        This function is automatically called by Scapy
        whenever a packet is captured.
        """

        if not self.is_running:
            return

        self.packet_callback(packet)

    def stop_capture(self) -> None:
        """
        Stops packet capture.

        Note:
        -----
        Scapy sniff() is blocking by default.
        In advanced implementations this method
        usually interacts with AsyncSniffer.
        """

        self.is_running = False

        print("Stopping capture...")


if __name__ == "__main__":

    from interface_manager import InterfaceManager

    def packet_processor(packet):
        """
        Example packet processing callback.
        """

        print(packet.summary())

    # Select interface
    manager = InterfaceManager()
    selected_interface = manager.select_interface()

    # Create sniffer
    sniffer = PacketSniffer(
        interface=selected_interface,
        packet_callback=packet_processor,
        filter_rule="ip"
    )

    # Start capture
    sniffer.start_capture()
