"""
=============================================================================

    This module contains tests for the functionalities in the qml module.

=============================================================================

    Copyright 2022 ACK Cyfronet AGH. All Rights Reserved.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

=============================================================================

    This work was supported by the EuroHPC PL project funded at the Smart Growth
    Operational Programme 2014-2020, Measure 4.2 under the grant agreement no.
    POIR.04.02.00-00-D014/20-00.

=============================================================================
"""

__author__ = "Tomasz Rybotycki"


import unittest
import abc
import os

import dill
import lightning.pytorch.utilities.seed

import pennylane as qml
from pennylane.operation import Operation
from pennylane.templates import StronglyEntanglingLayers
from pennylane.measurements import ExpectationMP
from pennylane import numpy as np

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from typing import Sequence, List, Type, Optional, Union, Tuple

from sklearn.datasets import (
    make_moons,
    make_regression,
    make_classification,
    load_digits,
)
from sklearn.metrics import rand_score

from numpy import isclose
from numpy.random import RandomState
from numpy.typing import NDArray
from aqmlator.qml import (
    QNNModel,
    QNNBinaryClassifier,
    QuantumKernelBinaryClassifier,
    QNNLinearRegression,
    QNNClassifier,
    RBMClustering,
)

from qiskit import IBMQ

from dwave.samplers import RandomSampler


class TestQNNModel(unittest.TestCase, abc.ABC):
    """
    A general `unittest.TestCase` class for QNN based QML models.
    """

    x: Sequence[Sequence[float]]
    y: Union[List[int], List[float]]

    model: QNNModel
    alternate_model: QNNModel

    def setUp(self) -> None:
        """
        Setup method for the `TestCase`. Should be overwritten by test classes.

        :note:
            TR: This is by default called before any test. One way to skip the test of this
            class if to set the skipping in the setUp. There may be a better way to do
            it though.
        """
        raise unittest.SkipTest("Skipping tests for abstract QNNModel class.")

    def tearDown(self) -> None:
        if os.path.isfile("qml_test.dil"):
            os.remove("qml_test.dil")

    @staticmethod
    def get_weights(model: torch.nn.Module) -> List[np.ndarray]:
        """
        Extract the weights from the given model.

        :param model:
            The model to extract the weights from.

        :return:
            The current weights of the model.
        """
        weights: List[np.ndarray] = []

        for _, param in model.named_parameters():
            weights.append(np.array(param.detach().numpy()))

        return weights

    def test_predict_run(self) -> None:
        """
        Tests if making predictions is possible.
        """
        self.model.predict(self.x)

    def test_fit_run(self) -> None:
        """
        Tests if the learning runs smoothly.
        """
        self.model.fit(self.x, self.y)

    def test_accuracy_increase(self) -> None:
        """
        Tests if the accuracy increases after short training.
        """
        initial_score: float = self.model.score(self.x, self.y)
        self.model.fit(self.x, self.y)
        final_score: float = self.model.score(self.x, self.y)
        self.assertTrue(
            final_score > initial_score,
            f"QNN Training: Initial score ({initial_score}) isn't worse than the final"
            f" score ({final_score})!",
        )

    def test_weights_change(self) -> None:
        """
        Tests if the weights change during the training.
        """
        initial_weights: Sequence[float] = self.model.weights
        self.model.fit(self.x, self.y)

        self.assertTrue(
            tuple(initial_weights) != tuple(self.model.weights),
            "Weights didn't change during the training!",
        )

    def test_results_dimensions(self) -> None:
        """
        Tests if the predictions have expected dimensions.
        """
        predictions: np.ndarray = self.model.predict(self.x)
        self.assertTrue(
            len(predictions) == len(self.x),
            f"Result dimensions are unexpected!({len(predictions)} != {len(self.x)}).",
        )

    def test_executions_number_growth(self) -> None:
        """
        Tests if the number of executions grows when the model is executed.
        """
        self.model.predict(self.x)
        self.assertTrue(
            self.model.n_executions() > 0, "The number of executions don't grow!"
        )

    def test_different_layers_predict_run(self) -> None:
        """
        Tests if making predictions is possible when different type of layers is used.
        """
        self.alternate_model.predict(self.x)

    def test_initial_serialization(self) -> None:
        """
        Tests if the model is serializable after the initialization.

        :note:
            This method can be (and is) used to check if the modified model is
            serializable.
        """

        model_score: float = self.model.score(self.x, self.y)

        with open("qml_test.dil", "wb") as f:
            dill.dump(self.model, f)

        with open("qml_test.dil", "rb") as f:
            loaded_model: QNNModel = dill.load(f)

        self.assertTrue(isclose(model_score, loaded_model.score(self.x, self.y)))

    def test_post_fit_serialization(self) -> None:
        """
        Tests if the model is serializable after fit.
        """
        self.model.fit(self.x, self.y)
        self.test_initial_serialization()

    def test_post_prediction_serialization(self) -> None:
        """
        Tests if the model is serializable after making prediction.
        """
        self.model.predict(self.x)
        self.test_initial_serialization()

    def test_torch_forward_run(self) -> None:
        """
        Tests if making predictions with torch classifier is possible.
        """
        model: torch.nn.Sequential = torch.nn.Sequential(self.model.get_torch_layer())
        model.forward(torch.tensor(self.x))

    def test_torch_results_dimension(self) -> None:
        """
        Tests if torch predictions have expected dimensions.
        """
        model: torch.nn.Sequential = torch.nn.Sequential(self.model.get_torch_layer())
        predictions: torch.Tensor = model.forward(torch.tensor(self.x))

        self.assertTrue(
            len(predictions) == len(self.x), "Torch predictions have unexpected shape."
        )

    def test_torch_different_layers_forward_run(self) -> None:
        """
        Tests if making predictions with torch is possible when different type of layers
        is used.
        """
        model: torch.nn.Sequential = torch.nn.Sequential(
            self.alternate_model.get_torch_layer()
        )
        model.forward(torch.tensor(self.x))


class TestQNNBinaryClassifier(TestQNNModel):
    """
    A `TestCase` class for the QNN-based binary classifier.
    """

    def setUp(self) -> None:
        """
        Sets up the tests.
        """
        # TR:   Changing the seed can cause problems with the `test_accuracy_increase`,
        #       as the number of training epochs is currently low for `seed = 1`.
        seed: int = 2
        noise: float = 0.1
        n_samples: int = 100
        accuracy_threshold: float = 0.85

        self.x, self.y = make_moons(
            n_samples=n_samples,
            shuffle=True,
            noise=noise,
            random_state=RandomState(seed),
        )

        for i in range(len(self.y)):
            if self.y[i] == 0:
                self.y[i] = -1

        n_qubits: int = 2

        dev: qml.Device = qml.device("lightning.qubit", wires=n_qubits)

        layers: List[Type[Operation]] = [
            StronglyEntanglingLayers
        ] * 3  # 3 StronglyEntanglingLayers

        alternate_layers: List[Type[Operation]] = [
            qml.templates.BasicEntanglerLayers
        ] * 2

        self.n_epochs: int = 2
        batch_size: int = 20

        self.model: QNNBinaryClassifier = QNNBinaryClassifier(
            wires=n_qubits,
            batch_size=batch_size,
            n_epochs=self.n_epochs,
            accuracy_threshold=accuracy_threshold,
            layers=layers,
            device=dev,
        )

        self.alternate_model: QNNBinaryClassifier = QNNBinaryClassifier(
            wires=n_qubits,
            batch_size=batch_size,
            n_epochs=self.n_epochs,
            accuracy_threshold=accuracy_threshold,
            layers=alternate_layers,
            device=dev,
        )


class TestQNNLinearRegressor(TestQNNModel):
    """
    A `TestCase` class for the QNN-based linear regressor.
    """

    def setUp(self) -> None:
        """
        Sets up the tests.
        """
        # TR:   Changing the seed can cause problems with the `test_accuracy_increase`,
        #       as the number of training epochs is currently low for `seed = 1`.
        seed: int = 2
        noise: float = 0.1
        n_samples: int = 100
        accuracy_threshold: float = 0.85

        self.x, self.y = make_regression(  # pylint: disable=unbalanced-tuple-unpacking
            n_samples=n_samples,
            n_features=2,
            shuffle=True,
            noise=noise,
            random_state=RandomState(seed),
        )

        n_qubits: int = 2
        dev: qml.Device = qml.device("lightning.qubit", wires=n_qubits)

        layers: List[Type[Operation]] = [
            StronglyEntanglingLayers
        ] * 3  # 3 StronglyEntanglingLayers

        alternate_layers: List[Type[Operation]] = [
            qml.templates.BasicEntanglerLayers
        ] * 2

        self.n_epochs: int = 3
        batch_size: int = 20

        self.model: QNNLinearRegression = QNNLinearRegression(
            wires=n_qubits,
            batch_size=batch_size,
            n_epochs=self.n_epochs,
            accuracy_threshold=accuracy_threshold,
            layers=layers,
            device=dev,
        )

        self.alternate_model: QNNLinearRegression = QNNLinearRegression(
            wires=n_qubits,
            batch_size=batch_size,
            n_epochs=self.n_epochs,
            accuracy_threshold=accuracy_threshold,
            layers=alternate_layers,
            device=dev,
        )


class TestQEKBinaryClassifier(unittest.TestCase):
    """
    A `TestCase` class for the QEK-based binary classifier.
    """

    def setUp(self) -> None:
        """
        Sets up the tests.
        """
        # TR:   Changing the seed can cause problems with the `test_accuracy_increase`,
        #       as the number of training epochs is currently minimal for `seed = 0`.
        seed: int = 0
        noise: float = 0.5
        n_samples: int = 15
        accuracy_threshold: float = 0.85

        self.x, self.y = make_moons(
            n_samples=n_samples,
            shuffle=True,
            noise=noise,
            random_state=RandomState(seed),
        )

        for i in range(len(self.y)):
            if self.y[i] == 0:
                self.y[i] = -1

        self.n_qubits: int = 2

        dev: qml.Device = qml.device("lightning.qubit", wires=self.n_qubits)

        layers: List[Type[Operation]] = [
            StronglyEntanglingLayers
        ] * 3  # 3 StronglyEntanglingLayers

        self.weights_length: int = 18

        alternate_layers: List[Type[Operation]] = [
            qml.templates.BasicEntanglerLayers
        ] * 3

        self.n_epochs: int = 1

        self.classifier: QuantumKernelBinaryClassifier = QuantumKernelBinaryClassifier(
            wires=self.n_qubits,
            n_epochs=self.n_epochs,
            accuracy_threshold=accuracy_threshold,
            layers=layers,
            device=dev,
        )

        self.alternate_classifier: QuantumKernelBinaryClassifier = (
            QuantumKernelBinaryClassifier(
                wires=self.n_qubits,
                n_epochs=self.n_epochs,
                accuracy_threshold=accuracy_threshold,
                layers=alternate_layers,
                device=dev,
            )
        )

    def tearDown(self) -> None:
        if os.path.isfile("qml_test.dil"):
            os.remove("qml_test.dil")

    def test_learning_and_predict_run(self) -> None:
        """
        Tests if fitting and making predictions is possible.
        """
        self.classifier.fit(self.x, self.y)
        self.classifier.predict(self.x)

    def test_accuracy_increase(self) -> None:
        """
        Tests if the accuracy increases after short training.
        """
        self.classifier.fit(self.x, self.y)
        initial_accuracy: float = self.classifier.score(self.x, self.y)

        self.classifier.n_epochs = 2  # Minimal required number in this setup.
        self.classifier.fit(self.x, self.y)
        accuracy: float = self.classifier.score(self.x, self.y)

        self.assertTrue(
            initial_accuracy < accuracy,
            f"Initial accuracy ({initial_accuracy}) didn't increase ({accuracy}) after "
            f"training.",
        )

    def test_weights_change(self) -> None:
        """
        Tests if the weights change during the training.
        """
        initial_weights: Sequence[float] = self.classifier.weights
        self.classifier.fit(self.x, self.y)

        self.assertTrue(
            tuple(initial_weights) != tuple(self.classifier.weights),
            "Weights didn't change during the training!",
        )

    def test_results_dimension(self) -> None:
        """
        Tests if the predictions have expected dimensions.
        """
        self.classifier.fit(self.x, self.y)
        predictions: np.ndarray = np.array(self.classifier.predict(self.x))
        self.assertTrue(
            predictions.shape == (len(self.x),),
            "QuantumKernelBinaryClassifier predictions have unexpected shape.",
        )

    def test_executions_number_growth(self) -> None:
        """
        Tests if the number of executions grows when the model is executed.
        """
        self.classifier.fit(self.x, self.y)
        self.classifier.predict(self.x)
        self.assertTrue(
            self.classifier.n_executions() > 0, "The number of executions don't grow!"
        )

    def test_different_layers_learning_and_predict_run(
        self,
    ) -> None:
        """
        Tests if making predictions is possible when different type of layers is used.
        """
        self.alternate_classifier.fit(self.x, self.y)
        self.alternate_classifier.predict(self.x)

    def test_transform_run(self) -> None:
        """
        Checks if `classifier.transform` runs.
        """
        self.classifier.fit(self.x, self.y)
        self.classifier.transform(self.x)

    def test_transform_dimension(self) -> None:
        """
        Checks if `classifier.transform` dimensions are as expected.
        """
        self.classifier.fit(self.x, self.y)
        mapped_x: List[ExpectationMP] = self.classifier.transform(self.x)

        self.assertTrue(
            len(mapped_x) == len(self.x),
            f"The results_reconstruction number is incorrect ({len(mapped_x)} != {len(self.x)})!",
        )

        for x in mapped_x:
            self.assertTrue(
                len(np.array(x)) == self.n_qubits,
                f"Dimension of the results_reconstruction is incorrect! ({len(np.array(x))} !="
                f" {self.n_qubits})",
            )

    def _test_serialization(self) -> None:
        """
        Tests if the model is serializable after the initialization.

        :note:
            This method can be (and is) used to check if the modified model is
            serializable.
        """

        model_score: float = self.classifier.score(self.x, self.y)

        with open("qml_test.dil", "wb") as f:
            dill.dump(self.classifier, f)

        with open("qml_test.dil", "rb") as f:
            loaded_model: QNNModel = dill.load(f)

        self.assertTrue(isclose(model_score, loaded_model.score(self.x, self.y)))

    def test_post_fit_serialization(self) -> None:
        """
        Tests if the model is serializable after fit.
        """
        self.classifier.fit(self.x, self.y)
        self._test_serialization()

    def test_post_fit_prediction_serialization(self) -> None:
        """
        Tests if the model is serializable after making prediction.
        """
        self.classifier.fit(self.x, self.y)
        self.classifier.predict(self.x)
        self._test_serialization()


class TestQuantumClassifier(unittest.TestCase):
    """
    A `TestCase` class for general quantum classifier.
    """

    def setUp(self) -> None:
        """
        Sets up the tests.
        """

        self.n_samples: int = 50
        seed: int = 0
        n_classes: int = 3
        n_epochs: int = 2
        batch_size: int = 10
        n_features: int = 2

        self.X: Sequence[Sequence[float]]
        self.y: Sequence[int]

        self.X, self.y = make_classification(
            n_samples=self.n_samples,
            n_features=n_features,
            n_classes=n_classes,
            n_redundant=0,
            n_clusters_per_class=1,
            random_state=RandomState(seed),
        )

        dev: qml.Device = qml.device("lightning.qubit", wires=n_features)

        classifiers: List[QNNBinaryClassifier] = [
            QNNBinaryClassifier(
                wires=n_features, batch_size=batch_size, n_epochs=n_epochs, device=dev
            )
            for _ in range(n_classes)
        ]

        self.classifier: QNNClassifier = QNNClassifier(
            wires=n_features,
            binary_classifiers=classifiers,
            n_classes=n_classes,
            device=dev,
        )

    def tearDown(self) -> None:
        if os.path.isfile("qml_test.dil"):
            os.remove("qml_test.dil")

    def test_predict_run(self) -> None:
        """
        Tests if making predictions is possible.
        """
        self.classifier.predict(self.X)

    def test_fit_run(self) -> None:
        """
        Tests if classifier fitting is possible.
        """
        self.classifier.fit(self.X, self.y)

    def test_results_dimensions(self) -> None:
        """
        Tests if the dimension of the results_reconstruction returned by the classifier is correct.
        """
        results: Sequence[int] = self.classifier.predict(self.X)
        self.assertTrue(len(results) == self.n_samples)

    def test_accuracy_increase(self) -> None:
        """
        Tests if the classifier accuracy increase after the training.
        """
        initial_accuracy: float = np.mean(
            [int(i == j) for i, j in zip(self.y, self.classifier.predict(self.X))]
        )
        self.classifier.fit(self.X, self.y)
        final_accuracy: float = np.mean(
            [int(i == j) for i, j in zip(self.classifier.predict(self.X), self.y)]
        )
        self.assertTrue(initial_accuracy < final_accuracy)

    def test_initial_serialization(self) -> None:
        """
        Tests if the model is serializable after the initialization.

        :note:
            This method can be (and is) used to check if the modified model is
            serializable.
        """

        model_score: float = self.classifier.score(self.X, self.y)

        with open("qml_test.dil", "wb") as f:
            dill.dump(self.classifier, f)

        with open("qml_test.dil", "rb") as f:
            loaded_model: QNNModel = dill.load(f)

        self.assertTrue(isclose(model_score, loaded_model.score(self.X, self.y)))

    def test_post_fit_serialization(self) -> None:
        """
        Tests if the model is serializable after fit.
        """
        self.classifier.fit(self.X, self.y)
        self.test_initial_serialization()

    def test_post_prediction_serialization(self) -> None:
        """
        Tests if the model is serializable after making prediction.
        """
        self.classifier.predict(self.X)
        self.test_initial_serialization()


# TODO TR: Think of a less general case for this class.
class TestIBMQDevicesHandling(unittest.TestCase):
    """
    A class for testing if the qml models work as intended on IBM devices.
    """

    def setUp(self) -> None:
        """
        Sets up the tests. Called before every test.
        """
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

        IBMQ.load_account()
        provider = IBMQ.get_provider(hub="ibm-q", group="open", project="main")

        backends = provider.backends()

        for i in range(len(backends)):
            if (
                "simulator" in str(backends[i]).lower()
                or backends[i].configuration().n_qubits < 3
            ):
                continue
            backend = backends[i]
            self.n_qubits: int = backend.configuration().n_qubits
            break

        config = backend.configuration()

        self.coupling_map: List[Sequence[int]] = config.coupling_map

        self.dev: qml.Device = qml.device(
            "qiskit.aer",
            wires=self.n_features,
            backend="aer_simulator_statevector",
        )

        self.coupled_dev = qml.device(
            "qiskit.aer",
            wires=self.n_features,
            backend="aer_simulator_statevector",
            coupling_map=self.coupling_map,
        )

        self.layers: List[Type[Operation]] = [
            StronglyEntanglingLayers
        ] * 3  # 3 StronglyEntanglingLayers

    def _proceed_with_qek_classifier_test(
        self,
        coupling_map: Optional[List[Sequence[int]]] = None,
        dev: Optional[qml.Device] = None,
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
        dev: Optional[qml.Device] = None,
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
        dev: Optional[qml.Device] = None,
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
        self._proceed_wth_qnn_classifier_test(
            dev=self.coupled_dev, coupling_map=self.coupling_map
        )

    def test_qnn_regressor_on_coupled_device(self) -> None:
        """
        Tests if the QNN regressor works correctly with the coupling map applied
        on the real IBMQ device simulator.
        """
        self._proceed_with_qnn_regressor_test(
            dev=self.coupled_dev, coupling_map=self.coupling_map
        )

    def test_qek_classifier_on_coupled_device(self) -> None:
        """
        Tests if the QNN classifier works correctly with the coupling map applied
        on the real IBMQ device simulator.
        """
        self._proceed_with_qek_classifier_test(
            dev=self.coupled_dev, coupling_map=self.coupling_map
        )


class TestRBMClustering(unittest.TestCase):
    """
    Tests for the RBMClustering class.
    """

    def setUp(self) -> None:
        """
        Sets up the test case.
        """
        lightning.pytorch.seed_everything(0, workers=True)  # Fix the seed.

        lbae_input_size: Tuple[int, ...] = (1, 8, 8)
        lbae_out_channels: int = 8
        n_layers: int = 2
        rbm_n_visible_neurons: int = 16
        rbm_n_hidden_neurons: int = 10

        n_epochs: int = 3
        batch_size: int = 10
        n_classes: int = 10

        X: NDArray[np.int]
        self.y: NDArray[np.int]

        X, self.y = load_digits(n_class=n_classes, return_X_y=True)
        X = X.reshape((1797, 1, 8, 8))

        self.X_tensor: Tensor = torch.Tensor(X)
        self.X_tensor /= 16  # Inputs have values from 0 to 16. Rescale to 0-1.

        # Use list instead of CustomDataset, as it implements everything that Dataset
        # is supposed to implement.
        dataset: List[Tuple[Tensor, Tensor]] = []

        for i in range(len(X)):
            dataset.append((self.X_tensor[i], self.y[i]))

        # Ignore mypy problem with the type of the dataset.
        self.data_loader: DataLoader[Tuple[Tensor, Tensor]] = DataLoader(
            dataset,  # type: ignore
            batch_size=batch_size,
            shuffle=True,
        )

        self.rbm_clustering: RBMClustering = RBMClustering(
            lbae_input_shape=lbae_input_size,
            lbae_out_channels=lbae_out_channels,
            lbae_n_layers=n_layers,
            rmb_n_visible_neurons=rbm_n_visible_neurons,
            rbm_n_hidden_neurons=rbm_n_hidden_neurons,
            n_epochs=n_epochs,
            rng=np.random.default_rng(seed=42),
            fireing_threshold=0.5,
        )

        self.x = self.X_tensor[0].view(1, 1, 8, 8)

    def tearDown(self) -> None:
        """
        Tears down the test case.
        """
        pass

    def _test_fit_run(self) -> None:
        """
        A common part of the tests for the fit method.
        """
        self.rbm_clustering.fit(self.data_loader)
        # self.assertTrue(True)

    def test_classical_clustering_fit_run(self) -> None:
        """
        Tests if the classically trained (using CD1 algorithm) RBMs fits without error.
        CD1 algorithm is used by default, when no sampler is specified.
        """
        self._test_fit_run()

    def test_sampler_clustering_fit_run(self) -> None:
        """
        Tests if the RBMs fits without error, when the sampler is specified.
        """
        sampler: RandomSampler = RandomSampler()
        self.rbm_clustering.sampler = sampler
        self._test_fit_run()

    def test_sampler_predict(self) -> None:
        """
        Tests if the RBMClustering predict method runs and returns binary values.
        """
        prediction: Tensor = self.rbm_clustering.predict(self.x)

        for val in prediction:
            self.assertTrue(val in (0, 1))

    def test_sampler_accuracy_increase(self) -> None:
        """
        Tests if the accuracy of the clustering increases after the (classical)
        training (which it should).
        """
        predictions: List[int] = []

        def simple_hash(t: Tensor) -> int:
            return sum(p * 2**i for i, p in enumerate(t))

        for x in self.X_tensor:
            predictions.append(
                simple_hash(self.rbm_clustering.predict(x.view(1, 1, 8, 8)))
            )

        initial_score: float = rand_score(self.y, predictions)

        predictions.clear()

        self.rbm_clustering.fit(self.data_loader)

        for x in self.X_tensor:
            predictions.append(
                simple_hash(self.rbm_clustering.predict(x.view(1, 1, 8, 8)))
            )

        final_score: float = rand_score(self.y, predictions)

        self.assertGreater(final_score, initial_score)
