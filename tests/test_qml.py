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
import pennylane
from pennylane.operation import Operation
from pennylane.templates import StronglyEntanglingLayers
from pennylane.measurements import ExpectationMP

import torch

from typing import Sequence, List, Tuple, Type
from sklearn.datasets import make_moons, make_regression, make_classification
from numpy import isclose
from numpy.random import RandomState
from aqmlator.qml import (
    QNNModel,
    QNNBinaryClassifier,
    QuantumKernelBinaryClassifier,
    QNNLinearRegression,
    QNNClassifier,
)

from pennylane import numpy as np


class TestQNNModel(unittest.TestCase, abc.ABC):
    """
    A general `unittest.TestCase` class for QNN based QML models.
    """

    def setUp(self) -> None:
        """
        Setup method for the `TestCase`. Should be overwritten by test classes.

        :note:
            TR: This is by default called before any test. One way to skip the test of this
            class if to set the skipping in the setUp. There may be a better way to do
            it though.
        """
        raise unittest.SkipTest

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

        for name, param in model.named_parameters():
            weights.append(np.array(param.detach().numpy()))

        return weights

    def test_predict_run(self) -> None:
        """
        Tests if making predictions is possible.
        """
        self.model.predict(self.x)
        self.assertTrue(True, "Predict crashed!")

    def test_fit_run(self) -> None:
        """
        Tests if the learning runs smoothly.
        """
        self.model.fit(self.x, self.y)
        self.assertTrue(True, "Fit crashed.")

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
        self.assertTrue(True, "Predict crashed!")

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
        self.assertTrue(True, "Torch forward crashed!")

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
        self.assertTrue(True, "Torch forward crashed!")


class TestQNNBinaryClassifier(TestQNNModel):
    """
    A `TestCase` class for the qnn module.
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

        self.x: Sequence[Sequence[float]]
        self.y: Sequence[int]

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

        dev: pennylane.Device = pennylane.device("lightning.qubit", wires=n_qubits)

        layers: List[Type[Operation]] = [
            StronglyEntanglingLayers
        ] * 3  # 3 StronglyEntanglingLayers
        layers_weights_shapes: List[Tuple[int, ...]] = [(1, n_qubits, 3)] * 3

        alternate_layers: List[Type[Operation]] = [
            pennylane.templates.BasicEntanglerLayers
        ] * 2
        alternate_layers_weights_shapes: List[Tuple[int, ...]] = [(1, n_qubits)] * 2

        self.n_epochs: int = 2
        batch_size: int = 20

        self.model: QNNBinaryClassifier = QNNBinaryClassifier(
            wires=n_qubits,
            batch_size=batch_size,
            n_epochs=self.n_epochs,
            accuracy_threshold=accuracy_threshold,
            layers=layers,
            layers_weights_shapes=layers_weights_shapes,
            device=dev,
        )

        self.alternate_model: QNNBinaryClassifier = QNNBinaryClassifier(
            wires=n_qubits,
            batch_size=batch_size,
            n_epochs=self.n_epochs,
            accuracy_threshold=accuracy_threshold,
            layers=alternate_layers,
            layers_weights_shapes=alternate_layers_weights_shapes,
            device=dev,
        )


class TestQNNLinearRegressor(TestQNNModel):
    """
    A `TestCase` class for the qnn module.
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

        self.x: Sequence[Sequence[float]]
        self.y: Sequence[int]

        self.x, self.y = make_regression(
            n_samples=n_samples,
            n_features=2,
            shuffle=True,
            noise=noise,
            random_state=RandomState(seed),
        )

        n_qubits: int = 2
        dev: pennylane.Device = pennylane.device("lightning.qubit", wires=n_qubits)

        layers: List[Type[Operation]] = [
            StronglyEntanglingLayers
        ] * 3  # 3 StronglyEntanglingLayers
        layers_weights_shapes: List[Tuple[int, ...]] = [(1, n_qubits, 3)] * 3

        alternate_layers: List[Type[Operation]] = [
            pennylane.templates.BasicEntanglerLayers
        ] * 2
        alternate_layers_weights_shapes: List[Tuple[int, ...]] = [(1, n_qubits)] * 2

        self.n_epochs: int = 3
        batch_size: int = 20

        self.model: QNNLinearRegression = QNNLinearRegression(
            wires=n_qubits,
            batch_size=batch_size,
            n_epochs=self.n_epochs,
            accuracy_threshold=accuracy_threshold,
            layers=layers,
            layers_weights_shapes=layers_weights_shapes,
            device=dev,
        )

        self.alternate_model: QNNLinearRegression = QNNLinearRegression(
            wires=n_qubits,
            batch_size=batch_size,
            n_epochs=self.n_epochs,
            accuracy_threshold=accuracy_threshold,
            layers=alternate_layers,
            layers_weights_shapes=alternate_layers_weights_shapes,
            device=dev,
        )


class TestQEKBinaryClassifier(unittest.TestCase):
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

        self.x: Sequence[Sequence[float]]
        self.y: Sequence[int]

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

        dev: pennylane.Device = pennylane.device("lightning.qubit", wires=self.n_qubits)

        layers: List[Type[Operation]] = [
            StronglyEntanglingLayers
        ] * 3  # 3 StronglyEntanglingLayers
        layers_weights_shapes: List[Tuple[int, ...]] = [(1, self.n_qubits, 3)] * 3

        self.weights_length: int = 18

        alternate_layers: List[Type[Operation]] = [
            pennylane.templates.BasicEntanglerLayers
        ] * 3
        alternate_layers_weights_shapes: List[Tuple[int, ...]] = [
            (1, self.n_qubits)
        ] * 2

        self.n_epochs: int = 1

        self.classifier: QuantumKernelBinaryClassifier = QuantumKernelBinaryClassifier(
            wires=self.n_qubits,
            n_epochs=self.n_epochs,
            accuracy_threshold=accuracy_threshold,
            layers=layers,
            layers_weights_shapes=layers_weights_shapes,
            device=dev,
        )

        self.alternate_classifier: QuantumKernelBinaryClassifier = (
            QuantumKernelBinaryClassifier(
                wires=self.n_qubits,
                n_epochs=self.n_epochs,
                accuracy_threshold=accuracy_threshold,
                layers=alternate_layers,
                layers_weights_shapes=alternate_layers_weights_shapes,
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
        self.assertTrue(True, "The fit crashed!")
        self.classifier.predict(self.x)
        self.assertTrue(True, "The prediction crashed!")

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
        predictions: np.ndarray = self.classifier.predict(self.x)
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
        self.assertTrue(True, "The learning crashed!")
        self.alternate_classifier.predict(self.x)
        self.assertTrue(True, "The predicting crashed!")

    def test_transform_run(self) -> None:
        """
        Checks if `classifier.transform` runs.
        """
        self.classifier.fit(self.x, self.y)
        self.classifier.transform(self.x)
        self.assertTrue(True, "Transform crashed!")

    def test_transform_dimension(self) -> None:
        """
        Checks if `classifier.transform` dimensions are as expected.
        """
        self.classifier.fit(self.x, self.y)
        mapped_x: List[ExpectationMP] = self.classifier.transform(self.x)

        self.assertTrue(
            len(mapped_x) == len(self.x),
            f"The results number is incorrect ({len(mapped_x)} != {len(self.x)})!",
        )

        for x in mapped_x:
            self.assertTrue(
                len(np.array(x)) == self.n_qubits,
                f"Dimension of the results is incorrect! ({len(np.array(x))} !="
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

        dev: pennylane.Device = pennylane.device("lightning.qubit", wires=n_features)

        classifiers: List[QNNBinaryClassifier] = [
            QNNBinaryClassifier(
                wires=n_features, batch_size=batch_size, n_epochs=n_epochs, device=dev
            )
            for _ in range(n_classes)
        ]

        self.classifier: QNNClassifier = QNNClassifier(
            wires=2, binary_classifiers=classifiers, n_classes=n_classes, device=dev
        )

    def tearDown(self) -> None:
        if os.path.isfile("qml_test.dil"):
            os.remove("qml_test.dil")

    def test_predict_run(self) -> None:
        """
        Tests if making predictions is possible.
        """
        self.classifier.predict(self.X)
        self.assertTrue(True)

    def test_fit_run(self) -> None:
        """
        Tests if classifier fitting is possible.
        """
        self.classifier.fit(self.X, self.y)
        self.assertTrue(True)

    def test_results_dimensions(self) -> None:
        """
        Tests if the dimension of the results returned by the classifier is correct.
        """
        results: Sequence[int] = self.classifier.predict(self.X)
        self.assertTrue(len(results) == self.n_samples)

    def test_accuracy_increase(self) -> None:
        """
        Tests if the classifier accuracy increase after the training.
        """
        initial_accuracy: float = (
            sum(self.classifier.predict(self.X) == self.y) / self.n_samples
        )
        self.classifier.fit(self.X, self.y)
        final_accuracy: float = (
            sum(self.classifier.predict(self.X) == self.y) / self.n_samples
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
