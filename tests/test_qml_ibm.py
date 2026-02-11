"""A separate test module for the IBMQ-related functionalities, as those tests take
forever to run.
"""

import os
import unittest
from typing import List, Optional, Sequence, Type

import pennylane as qml
import warnings
from numpy.random import RandomState
from pennylane.operation import Operation
from pennylane.templates import StronglyEntanglingLayers
from qiskit_ibm_runtime import QiskitRuntimeService
from sklearn.datasets import (
    make_classification,
    make_regression,
)

from aqmlator.qml import (
    QNNBinaryClassifier,
    QNNLinearRegression,
    QuantumKernelBinaryClassifier,
)


# TODO TR: Think of a less general case for this class.
class TestIBMQDevicesHandling(unittest.TestCase):
    """
    A class for testing if the qml models work as intended on IBM devices.
    """

    def setUp(self) -> None:
        """
        Sets up the tests. Called before every test.
        """
        # self.skipTest("Skip IBMQ")
        # TR: In case Qiskit and PennyLane versions are compatible. Latest aren't.
        # warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore")

        n_samples: int = 50
        seed: int = 0

        self.n_features: int = 3
        self.n_classes: int = 2
        self.noise: float = 0.1
        self.batch_size: int = n_samples // 5
        self.n_epochs: int = 1
        self.accuracy_threshold: float = 0.8

        self.class_X: Sequence[Sequence[float]]
        self.class_y: Sequence[int]

        self.class_X, self.class_y = make_classification(
            n_samples=n_samples,
            n_features=self.n_features,
            n_classes=self.n_classes,
            n_redundant=0,
            n_clusters_per_class=1,
            random_state=RandomState(seed),
        )

        self.regression_X: Sequence[Sequence[float]]
        self.regression_y: Sequence[float]

        (  # pylint: disable=unbalanced-tuple-unpacking
            self.regression_X,
            self.regression_y,
        ) = make_regression(
            n_samples=n_samples,
            n_features=self.n_features,
            shuffle=True,
            noise=self.noise,
            random_state=RandomState(seed),
        )

        service = QiskitRuntimeService(
            channel="ibm_quantum_platform",
            token=os.environ["IBMQ_TOKEN"],
            instance=os.environ["IBMQ_CRN"],
        )

        backends = service.backends()

        for i in range(len(backends)):
            if "simulator" in str(backends[i]).lower() or backends[i].configuration().n_qubits < 3:
                continue
            backend = backends[i]
            self.n_qubits: int = backend.configuration().n_qubits
            break

        config = backend.configuration()

        self.coupling_map: List[Sequence[int]] = config.coupling_map

        self.dev: qml.devices.Device = qml.device(
            "qiskit.aer",
            wires=self.n_features,
        )

        self.coupled_dev: qml.device_api.Device = qml.device(
            "qiskit.aer",
            wires=self.n_features,
            coupling_map=self.coupling_map,
            basis_gates=config.to_dict()[
                "basis_gates"
            ],  # To remove the issue of 3-qubit gates in qiskit.aer basis_gates
        )

        self.layers: List[Type[Operation]] = [StronglyEntanglingLayers] * 3  # 3 StronglyEntanglingLayers

    def _proceed_with_qek_classifier_test(
        self,
        coupling_map: Optional[List[Sequence[int]]] = None,
        dev: Optional[qml.devices.Device] = None,
    ) -> None:
        """
        A common part of all the QEK Classifier-related tests. Test is passed if the
        fitting don't crash.

        :param coupling_map:
            A coupling map to be applied when applying the VQC.
        :param dev:
            A device to run the VQC on.
        """
        if not dev:
            dev = self.dev

        qek_classifier: QuantumKernelBinaryClassifier = QuantumKernelBinaryClassifier(
            wires=self.n_features,
            n_epochs=self.n_epochs,
            accuracy_threshold=self.accuracy_threshold,
            layers=self.layers,
            device=dev,
            coupling_map=coupling_map,
        )
        qek_classifier.fit(self.class_X, self.class_y)

    def _proceed_with_qnn_regressor_test(
        self,
        coupling_map: Optional[List[Sequence[int]]] = None,
        dev: Optional[qml.devices.Device] = None,
    ) -> None:
        """
        A common part of all the QNN Regressor-related tests. Test is passed if the
        fitting don't crash.

        :param coupling_map:
            A coupling map to be applied when applying the VQC.
        :param dev:
            A device to run the VQC on.
        """
        if not dev:
            dev = self.dev

        qnn_regressor: QNNLinearRegression = QNNLinearRegression(
            wires=self.n_features,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            accuracy_threshold=self.accuracy_threshold,
            layers=self.layers,
            device=dev,
            coupling_map=coupling_map,
        )
        qnn_regressor.fit(self.regression_X, self.regression_y)

    def _proceed_wth_qnn_classifier_test(
        self,
        coupling_map: Optional[List[Sequence[int]]] = None,
        dev: Optional[qml.devices.Device] = None,
    ) -> None:
        """
        A common part of all the QNN Classifier-related tests. Test is passed if the
        fitting don't crash.

        :param coupling_map:
            A coupling map to be applied when applying the VQC.
        :param dev:
            A device to run the VQC on.
        """
        if not dev:
            dev = self.dev

        qnn_classifier: QNNBinaryClassifier = QNNBinaryClassifier(
            wires=self.n_features,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            accuracy_threshold=self.accuracy_threshold,
            layers=self.layers,
            device=dev,
            coupling_map=coupling_map,
        )

        qnn_classifier.fit(self.class_X, self.class_y)

    def test_qek_classifier_on_qiskit_simulator(self) -> None:
        """
        Tests if the QEK classifier works correctly on the unconstrained IBMQ device
        simulator.
        """
        self._proceed_with_qek_classifier_test()

    def test_qnn_classifier_on_qiskit_simulator(self) -> None:
        """
        Tests if the QNN classifier works correctly on the unconstrained IBMQ device
        simulator.
        """
        self._proceed_wth_qnn_classifier_test()

    def test_qnn_regressor_on_qiskit_simulator(self) -> None:
        """
        Tests if the QNN regressor works correctly on the unconstrained IBMQ device
        simulator.
        """
        self._proceed_with_qnn_regressor_test()

    def test_qek_classifier_with_coupling(self) -> None:
        """
        Tests if the QEK classifier works correctly with the coupling map applied
        on the unconstrained IBMQ device simulator.
        """
        self._proceed_with_qek_classifier_test(self.coupling_map)

    def test_qnn_classifier_with_coupling(self) -> None:
        """
        Tests if the QNN classifier works correctly with the coupling map applied
        on the unconstrained IBMQ device simulator.
        """
        self._proceed_wth_qnn_classifier_test(self.coupling_map)

    def test_qnn_regressor_with_coupling(self) -> None:
        """
        Tests if the QNN regressor works correctly with the coupling map applied
        on the unconstrained IBMQ device simulator.
        """
        self._proceed_with_qnn_regressor_test(self.coupling_map)

    def test_qnn_classifier_on_coupled_device(self) -> None:
        """
        Tests if the QNN classifier works correctly with the coupling map applied
        on the real IBMQ device simulator.
        """
        self._proceed_wth_qnn_classifier_test(dev=self.coupled_dev, coupling_map=self.coupling_map)

    def test_qnn_regressor_on_coupled_device(self) -> None:
        """
        Tests if the QNN regressor works correctly with the coupling map applied
        on the real IBMQ device simulator.
        """
        self._proceed_with_qnn_regressor_test(dev=self.coupled_dev, coupling_map=self.coupling_map)

    def test_qek_classifier_on_coupled_device(self) -> None:
        """
        Tests if the QNN classifier works correctly with the coupling map applied
        on the real IBMQ device simulator.
        """
        self._proceed_with_qek_classifier_test(dev=self.coupled_dev, coupling_map=self.coupling_map)
