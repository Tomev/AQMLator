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
import torch

from typing import Sequence, List
from sklearn.datasets import make_moons
from numpy.random import RandomState
from aqmlator.qnn import QNNBinaryClassifier

from pennylane import numpy as np


class TestQNN(unittest.TestCase):
    """
    A `TestCase` class for the qnn module.
    """

    def setUp(self):
        """
        Sets up the tests.
        """
        seed: int = 42
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
        n_layers: int = 3
        self.n_epochs: int = 2
        batch_size: int = 20

        self.classifier: QNNBinaryClassifier = QNNBinaryClassifier(
            n_qubits=n_qubits,
            n_layers=n_layers,
            batch_size=batch_size,
            n_epochs=self.n_epochs,
            accuracy_threshold=accuracy_threshold,
        )

    @staticmethod
    def get_weights(model: torch.nn.Module) -> List[np.ndarray]:
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

        :return:
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
