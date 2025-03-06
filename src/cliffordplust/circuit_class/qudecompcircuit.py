from typing import List

from cliffordplust.circuit_class import QudecompGate
from cliffordplust.exact_synthesis import *


class QudecompCircuit:
    def __init__(self, gates: List["QudecompGate"]):
        """
        Initialize the QuantumCircuit with a list of quantum gates.

        :param gates: List of quantum gates
        """
        self.gates = gates

    def add_gate(self, gate: "QudecompGate") -> None:
        """
        Add a quantum gate to the circuit.

        :param gate: A quantum gate to be added
        """
        self.gates.append(gate)

    def remove_gate(self, gate: "QudecompGate") -> None:
        """
        Remove a quantum gate from the circuit.

        :param gate: A quantum gate to be removed
        """
        self.gates.remove(gate)

    def __repr__(self) -> str:
        """
        Return a string representation of the quantum circuit.
        """
        return f"QudecompCircuit({self.gates})"
