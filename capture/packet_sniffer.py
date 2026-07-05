from typing import Callable, Optional

from scapy.all import AsyncSniffer


class PacketSniffer:
    """
    Handles live packet capture from a selected network interface.

    Responsibilities
    ----------------
    - Start live packet capture
    - Stop packet capture
    - Forward captured packets to callback
    - Maintain capture statistics
    """

    def __init__(
        self,
        interface: str,
        packet_callback: Callable,
        packet_count: Optional[int] = None,
        timeout: Optional[int] = None,
        filter_rule: Optional[str] = "tcp or udp",
    ):
        """
        Parameters
        ----------
        interface
            Network interface.

        packet_callback
            Function called for every captured packet.

        packet_count
            Maximum packets to capture.
            None = unlimited.

        timeout
            Capture timeout in seconds.
            None = unlimited.

        filter_rule
            BPF capture filter.
        """

        self.interface = interface

        self.packet_callback = packet_callback

        self.packet_count = packet_count

        self.timeout = timeout

        self.filter_rule = filter_rule

        self.is_running = False

        self.packet_counter = 0

        self.sniffer = None

    # ------------------------------------------------------------------
    # Start Capture
    # ------------------------------------------------------------------

    def start_capture(self) -> None:

        if self.is_running:

            print("Capture already running.")

            return

        print("=" * 60)
        print("LIVE PACKET CAPTURE")
        print("=" * 60)

        print(f"Interface : {self.interface}")
        print(f"Filter    : {self.filter_rule}")

        if self.packet_count:

            print(f"Packet Limit : {self.packet_count}")

        if self.timeout:

            print(f"Timeout      : {self.timeout} sec")

        print()

        self.packet_counter = 0

        self.is_running = True

        sniffer_arguments = {

            "iface": self.interface,

            "prn": self.handle_packet,

            "store": False,
        }

        if self.filter_rule is not None:

            sniffer_arguments["filter"] = self.filter_rule

        if self.packet_count is not None:

            sniffer_arguments["count"] = self.packet_count

        self.sniffer = AsyncSniffer(

            **sniffer_arguments

        )

        try:

            self.sniffer.start()

            if self.timeout is None:

                while self.is_running:

                    self.sniffer.join(0.5)

            else:

                self.sniffer.join(timeout=self.timeout)

                self.stop_capture()

        except KeyboardInterrupt:

            print("\nStopping capture...")

            self.stop_capture()

        except Exception as error:

            print(f"\nCapture Error : {error}")

            self.stop_capture()

    # ------------------------------------------------------------------
    # Packet Handler
    # ------------------------------------------------------------------

    def handle_packet(
        self,
        packet,
    ) -> None:

        if not self.is_running:

            return

        self.packet_counter += 1

        self.packet_callback(packet)

    # ------------------------------------------------------------------
    # Stop Capture
    # ------------------------------------------------------------------

    def stop_capture(self):

        if not self.is_running:

            return

        self.is_running = False

        if self.sniffer is not None:

            self.sniffer.stop()

        print()

        print("=" * 60)
        print("CAPTURE FINISHED")
        print("=" * 60)

        print(
            f"Packets Captured : "
            f"{self.packet_counter}"
        )

        print("=" * 60)
    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self):

        return {

            "interface": self.interface,

            "packets_captured": self.packet_counter,

            "running": self.is_running,
        }


# ==========================================================================
# Example
# ==========================================================================

if __name__ == "__main__":

    from interface_manager import InterfaceManager

    def packet_processor(packet):

        print(packet.summary())

    manager = InterfaceManager()

    interface = manager.select_interface()

    sniffer = PacketSniffer(

        interface=interface,

        packet_callback=packet_processor,

        filter_rule="tcp or udp",

    )

    sniffer.start_capture()