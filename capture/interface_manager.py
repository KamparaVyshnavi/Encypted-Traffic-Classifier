import psutil
from typing import List, Optional


class InterfaceManager:
    """
    Handles:
    - discovering available network interfaces
    - validating interfaces
    - selecting capture interface
    """

    def __init__(self):
        self.interfaces = self.get_available_interfaces()

    def get_available_interfaces(self) -> List[str]:
        """
        Returns all available network interfaces.
        """
        return list(psutil.net_if_addrs().keys())

    def display_interfaces(self) -> None:
        """
        Prints all available interfaces.
        """
        print("\nAvailable Network Interfaces:\n")

        for index, interface in enumerate(self.interfaces, start=1):
            print(f"{index}. {interface}")

    def validate_interface(self, interface_name: str) -> bool:
        """
        Checks whether the given interface exists.
        """
        return interface_name in self.interfaces

    def select_interface(self, interface_name: Optional[str] = None) -> str:
        """
        Selects a network interface.

        If interface_name is provided:
            - validates and returns it.

        Otherwise:
            - asks user to select interactively.
        """

        if interface_name:
            if self.validate_interface(interface_name):
                return interface_name
            else:
                raise ValueError(f"Invalid interface: {interface_name}")

        self.display_interfaces()

        while True:
            try:
                choice = int(input("\nSelect interface number: "))

                if 1 <= choice <= len(self.interfaces):
                    selected_interface = self.interfaces[choice - 1]
                    print(f"\nSelected Interface: {selected_interface}")
                    return selected_interface

                print("Invalid choice. Try again.")

            except ValueError:
                print("Please enter a valid number.")


if __name__ == "__main__":
    manager = InterfaceManager()

    selected = manager.select_interface()

    print(f"\nUsing interface: {selected}")
