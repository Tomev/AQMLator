"""TODO(TR): Document this module"""

import numpy as np
import pennylane as qml
from pennylane.math import shape
from pennylane.templates import SimplifiedTwoDesign as STD
from pennylane.wires import Wires
from torch import Tensor


class BellmanLayer(qml.operation.Operation):
    """TODO(TR): Add class docstring"""

    n_wires: int | None = None  # Defaults to None, and uses all wires
    grad_method = None

    def __init__(self, weights: Tensor, wires: Wires, id: str | None = None) -> None:  # pylint: disable=redefined-builtin
        # TR: I can add some additional checks here.
        super().__init__(weights, wires=wires, id=id)

    @staticmethod
    def compute_decomposition(weights: Tensor, wires: Wires) -> list[qml.operation.Operation]:
        n_layers: int = shape(weights)[0]
        op_list: list[qml.operation.Operation] = []

        for layer in range(n_layers):
            op_list.append(qml.H(wires=wires[0]))

            for i in range(len(wires) - 1):
                op_list.append(qml.CNOT(wires=[wires[i], wires[i + 1]]))

            for i in range(len(wires)):
                op_list.append(qml.RY(phi=weights[layer][i], wires=wires[i]))

            for i in range(len(wires) - 1, 0, -1):
                op_list.append(qml.CNOT(wires=[wires[i - 1], wires[i]]))

        return op_list

    @staticmethod
    def shape(n_layers: int, n_wires: int) -> tuple[int, int]:
        r"""Returns a list of shapes for the 2 parameter tensors.

        Args:
            n_layers (int): number of layers
            n_wires (int): number of wires

        Returns:
            list[tuple[int]]: list of shapes
        """
        return (n_layers, n_wires)


class SimplifiedTwoDesign:
    """TODO(TR): Add class docstring"""

    def __init__(self, weights: Tensor, wires: Wires) -> None:
        STD(weights=weights, wires=wires, initial_layer_weights=np.zeros_like(wires))

    @staticmethod
    def compute_decomposition(weights: Tensor, wires: Wires) -> list[qml.operation.Operation]:
        return STD.compute_decomposition(
            weights=weights,
            wires=wires,
            initial_layer_weights=np.zeros_like(wires),
        )

    @staticmethod
    def __call__(weights: Tensor, wires: Wires) -> None:
        STD(weights=weights, wires=wires, initial_layer_weights=np.zeros_like(wires))

    @staticmethod
    def shape(n_layers: int, n_wires: int) -> tuple[int, ...]:
        return STD.shape(n_layers=n_layers, n_wires=n_wires)[1]  # Return only shape of weights, skip initial_weights
