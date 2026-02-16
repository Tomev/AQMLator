"""TODO(TR): Document this module"""

from typing import Any

import pennylane as qml
from pennylane.math import shape
from pennylane.wires import Wires


class BellmanLayer(qml.operation.Operation):
    n_wires: int | None = None  # Defaults to None, and uses all wires
    grad_method = None

    def __init__(self, weights, wires, id=None):
        # TR: I can add some additional checks here.
        super().__init__(weights, wires=wires, id=id)

    @staticmethod
    def compute_decomposition(weights, wires: Wires, **hyperparameters: dict[str, Any]) -> list[qml.operation.Operator]:
        n_layers: int = shape(weights)[0]
        op_list = []

        for layer in range(n_layers):
            op_list.append(qml.H(wires=wires[0]))

            for i in range(len(wires) - 1):
                op_list.append([qml.CNOT(wires=[wires[i], wires[i + 1]])])

            for i in range(len(wires)):
                op_list.append(qml.RY(phi=weights[layer][i], wires=wires[i]))

            for i in range(len(wires) - 1, 0, -1):
                op_list.append([qml.CNOT(wires=[wires[i - 1], wires[i]])])

        return op_list

    @staticmethod
    def shape(n_layers, n_wires):
        r"""Returns a list of shapes for the 2 parameter tensors.

        Args:
            n_layers (int): number of layers
            n_wires (int): number of wires

        Returns:
            list[tuple[int]]: list of shapes
        """
        return [(n_wires,), (n_layers, n_wires)]
