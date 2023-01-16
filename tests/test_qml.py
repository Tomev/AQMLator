"""
=============================================================================

    This module contains ...

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

import pennylane
from pennylane.operation import Operation
from pennylane.templates import StronglyEntanglingLayers
from pennylane.measurements import ExpectationMP

import torch


from typing import Sequence, List, Tuple
from sklearn.datasets import make_moons
from numpy.random import RandomState
from aqmlator.qml import QNNBinaryClassifier, QuantumKernelBinaryClassifier

from pennylane import numpy as np


class TestQNN(unittest.TestCase):
    """
    A `TestCase` class for the qnn module.
    """

    def setUp(self) -> None:
        """
        Sets up the tests.
        """
        # TR:   Changing the seed can cause problems with the `test_accuracy_increase`,
        #       as the number of training epochs is currently low for `seed = 1`.
        seed: int = 1
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

        n_qubits: int = 2

        layers: List[Operation] = [
            StronglyEntanglingLayers
        ] * 3  # 3 StronglyEntanglingLayers
        layers_weights_shapes: List[Tuple[int, ...]] = [(1, n_qubits, 3)] * 3

        alternate_layers: List[Operation] = [
            pennylane.templates.BasicEntanglerLayers
        ] * 2
        alternate_layers_weights_shapes: List[Tuple[int, ...]] = [(1, n_qubits)] * 2

        self.n_epochs: int = 2
        batch_size: int = 20

        self.classifier: QNNBinaryClassifier = QNNBinaryClassifier(
            n_qubits=n_qubits,
            batch_size=batch_size,
            n_epochs=self.n_epochs,
            accuracy_threshold=accuracy_threshold,
            layers=layers,
            layers_weights_shapes=layers_weights_shapes,
        )

        self.alternate_classifier: QNNBinaryClassifier = QNNBinaryClassifier(
            n_qubits=n_qubits,
            batch_size=batch_size,
            n_epochs=self.n_epochs,
            accuracy_threshold=accuracy_threshold,
            layers=alternate_layers,
            layers_weights_shapes=alternate_layers_weights_shapes,
        )

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

    def test_forward_run(self) -> None:
        """
        Tests if making predictions is possible.
        """
        self.classifier.predict(self.x)
        self.assertTrue(True, "The forward crashed!")

    def test_torch_forward_run(self) -> None:
        """
        Tests if making predictions with torch classifier is possible.
        """
        model: torch.nn.Sequential = torch.nn.Sequential(
            self.classifier.get_torch_layer()
        )
        model.forward(torch.tensor(self.x))
        self.assertTrue(True, "The torch forward crashed!")

    def test_learning_run(self) -> None:
        """
        Tests if the learning runs smoothly.
        """
        self.classifier.fit(self.x, self.y)
        self.assertTrue(True, "The learning crashed.")

    def test_torch_learning_run(self) -> None:
        """
        Tests if learning using PyTorch runs smoothly.

        The code is taken from the PyTorch docs:
        https://pytorch.org/docs/stable/optim.html
        """
        model: torch.nn.Sequential = torch.nn.Sequential(
            self.classifier.get_torch_layer()
        )

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        tensor_x: torch.Tensor = torch.tensor(self.x)
        tensor_y: torch.Tensor = torch.tensor(self.y)

        model.train()

        # Training
        optimizer.zero_grad()
        output: torch.Tensor = model(tensor_x)
        loss: torch.Tensor = torch.nn.functional.l1_loss(output, tensor_y)
        loss.backward()
        optimizer.step()

        self.assertTrue(True, "The torch learning crashed.")

    def test_accuracy_increase(self) -> None:
        """
        Tests if the accuracy increases after short training.
        """
        initial_score: float = self.classifier.score(self.x, self.y)
        self.classifier.fit(self.x, self.y)
        final_score: float = self.classifier.score(self.x, self.y)
        self.assertTrue(
            final_score > initial_score,
            f"QNN Training: Initial score ({initial_score}) is greater than the final"
            f" score ({final_score})!",
        )

    def test_torch_accuracy_increase(self) -> None:
        """
        Tests if the torch accuracy increases after short training.
        """
        model: torch.nn.Sequential = torch.nn.Sequential(
            self.classifier.get_torch_layer()
        )

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        tensor_x: torch.Tensor = torch.tensor(self.x)
        tensor_y: torch.Tensor = torch.tensor(self.y)

        output: torch.Tensor = model(tensor_x)
        initial_loss: torch.Tensor = np.array(
            torch.nn.functional.l1_loss(output, tensor_y).detach().numpy()
        )

        model.train()

        # Training
        for _ in range(self.n_epochs):
            optimizer.zero_grad()
            output = model(tensor_x)
            loss: torch.Tensor = torch.nn.functional.l1_loss(output, tensor_y)
            loss.backward()
            optimizer.step()

        final_loss: float = loss.detach().numpy()

        self.assertTrue(
            final_loss < initial_loss, "The torch learning decreases the accuracy!."
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

    def test_torch_weights_change(self) -> None:
        """
        Tests if the weights change during the torch training.
        """
        model: torch.nn.Sequential = torch.nn.Sequential(
            self.classifier.get_torch_layer()
        )

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        tensor_x: torch.Tensor = torch.tensor(self.x)
        tensor_y: torch.Tensor = torch.tensor(self.y)

        model.train()

        initial_weights: List[np.ndarray] = self.get_weights(model)

        # Training
        optimizer.zero_grad()
        output: torch.Tensor = model(tensor_x)
        loss: torch.Tensor = torch.nn.functional.l1_loss(output, tensor_y)
        loss.backward()
        optimizer.step()

        trained_weights: List[np.ndarray] = self.get_weights(model)

        self.assertTrue(
            len(trained_weights) == len(initial_weights),
            "Weights changed dimensions during torch training.",
        )

        for i in range(len(trained_weights)):
            self.assertTrue(
                (trained_weights[i] != initial_weights[i]).any(),
                "Weights didn't change during torch training!",
            )

    def test_results_dimensions(self) -> None:
        """
        Tests if the predictions have expected dimensions.
        """
        predictions: np.ndarray = self.classifier.predict(self.x)
        self.assertTrue(
            predictions.shape == (len(self.x),),
            "QNNBinaryClassifier predictions have unexpected shape.",
        )

    def test_torch_results_dimension(self) -> None:
        """
        Tests if torch predictions have expected dimensions.
        """
        model: torch.nn.Sequential = torch.nn.Sequential(
            self.classifier.get_torch_layer()
        )
        predictions: torch.Tensor = model.forward(torch.tensor(self.x))

        self.assertTrue(
            len(predictions) == len(self.x), "Torch predictions have unexpected shape."
        )

    def test_executions_number_growth(self) -> None:
        """
        Tests if the number of executions grows when the model is executed.
        """
        self.classifier.predict(self.x)
        self.assertTrue(
            self.classifier.n_executions() > 0, "The number of executions don't grow!"
        )

    def test_different_layers_forward_run(self) -> None:
        """
        Tests if making predictions is possible when different type of layers is used.
        """
        self.alternate_classifier.predict(self.x)
        self.assertTrue(True, "The forward crashed!")

    def test_different_layers_torch_forward_run(self) -> None:
        """
        Tests if making predictions with torch is possible when different type of layers
        is used.
        """
        model: torch.nn.Sequential = torch.nn.Sequential(
            self.alternate_classifier.get_torch_layer()
        )
        model.forward(torch.tensor(self.x))
        self.assertTrue(True, "The torch forward crashed!")


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

        layers: List[Operation] = [
            StronglyEntanglingLayers
        ] * 3  # 3 StronglyEntanglingLayers
        layers_weights_shapes: List[Tuple[int, ...]] = [(1, self.n_qubits, 3)] * 3

        self.weights_length: int = 18

        alternate_layers: List[Operation] = [
            pennylane.templates.BasicEntanglerLayers
        ] * 3
        alternate_layers_weights_shapes: List[Tuple[int, ...]] = [
            (1, self.n_qubits)
        ] * 2

        self.n_epochs: int = 1

        self.classifier: QuantumKernelBinaryClassifier = QuantumKernelBinaryClassifier(
            n_qubits=self.n_qubits,
            n_epochs=self.n_epochs,
            accuracy_threshold=accuracy_threshold,
            layers=layers,
            layers_weights_shapes=layers_weights_shapes,
        )

        self.alternate_classifier: QuantumKernelBinaryClassifier = (
            QuantumKernelBinaryClassifier(
                n_qubits=self.n_qubits,
                n_epochs=self.n_epochs,
                accuracy_threshold=accuracy_threshold,
                layers=alternate_layers,
                layers_weights_shapes=alternate_layers_weights_shapes,
            )
        )

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
            f"Initial accuracy ({initial_accuracy}) didn't increase ({accuracy}) after training.",
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
                f"Dimension of the results is incorrect! ({len(np.array(x))} != {self.n_qubits})",
            )
