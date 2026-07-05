"""
===============================================================================
Packet Sniffer + Interface Manager Test
===============================================================================

Tests

✓ Interface discovery
✓ Interface selection
✓ Packet capture
✓ Callback execution
✓ Statistics
✓ Graceful shutdown

Stop capture using Ctrl+C.
===============================================================================
"""

import time

from capture.interface_manager import InterfaceManager
from capture.packet_sniffer import PacketSniffer


# =============================================================================
# Utility
# =============================================================================

def check(condition, message):

    if not condition:
        raise AssertionError(f"✗ {message}")

    print(f"✓ {message}")


# =============================================================================
# Packet Callback
# =============================================================================

captured_packets = []


def packet_callback(packet):

    captured_packets.append(packet)

    print(f"[{len(captured_packets):04d}] {packet.summary()}")


# =============================================================================
# Main Test
# =============================================================================

def main():

    print("=" * 70)
    print("PACKET SNIFFER TEST")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Interface Discovery
    # -------------------------------------------------------------------------

    manager = InterfaceManager()

    interfaces = manager.get_available_interfaces()

    check(len(interfaces) > 0, "Interfaces discovered")

    print()

    manager.display_interfaces()

    print()

    interface = manager.select_interface()

    check(manager.validate_interface(interface),
          "Interface validation")

    # -------------------------------------------------------------------------
    # Sniffer Creation
    # -------------------------------------------------------------------------

    sniffer = PacketSniffer(

        interface=interface,

        packet_callback=packet_callback,

        timeout=20,

        filter_rule="tcp or udp",
    )

    check(sniffer.interface == interface,
          "Sniffer created")

    print()

    print("=" * 70)
    print("Generating some network traffic is recommended.")
    print("Examples:")
    print("  • Open a website")
    print("  • Search something on Google")
    print("  • Play a YouTube video")
    print("  • Ping any website")
    print("=" * 70)

    print()

    # -------------------------------------------------------------------------
    # Capture
    # -------------------------------------------------------------------------

    start = time.perf_counter()

    sniffer.start_capture()

    end = time.perf_counter()

    elapsed = end - start

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    stats = sniffer.get_statistics()

    print()

    print("=" * 70)
    print("CAPTURE STATISTICS")
    print("=" * 70)

    print(f"Capture Time     : {elapsed:.2f} sec")
    print(f"Packets Captured : {stats['packets_captured']}")
    print(f"Running          : {stats['running']}")

    print()

    check(
        stats["packets_captured"] >= len(captured_packets),
        "Packet statistics valid",
    )

    check(
        not stats["running"],
        "Capture stopped successfully",
    )

    check(
        len(captured_packets) > 0,
        "Packets received",
    )

    print()

    print("=" * 70)
    print("ALL TESTS PASSED SUCCESSFULLY")
    print("=" * 70)


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":

    main()