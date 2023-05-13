"""
=============================================================================

    This module contains tests for the functionalities in the tuner module.

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

from typing import Sequence

from aqmlator.tuner import ModelFinder, HyperparameterTuner, MLTaskType
from aqmlator.qml import QNNBinaryClassifier
from sklearn.datasets import make_moons, make_regression, make_classification
from numpy.random import RandomState
import pennylane as qml


class TestModelFinder(unittest.TestCase):
    """
    This is a `TestCase` for the `ModelFinder` class.
    """

    def setUp(self) -> None:
        """
        Sets up the tests.

        :Note:
            There's no return value and only a single method to test, so we strive
            to be as minimalistic as possible.
        """

        x: Sequence[Sequence[float]]
        y: Sequence[int]

        x, y = make_moons(
            n_samples=10,
            shuffle=True,
            noise=0.1,
            random_state=RandomState(0),
        )

        reg_x, reg_y = make_regression(
            n_samples=10,
            n_features=2,
            n_informative=2,
            n_targets=1,
            random_state=RandomState(0),
        )

        cls_x, cls_y = make_classification(
            n_samples=10,
            n_features=2,
            n_informative=2,
            n_redundant=0,
            n_repeated=0,
            n_classes=3,
            n_clusters_per_class=1,
            random_state=RandomState(0),
        )

        n_qubits: int = 2
        n_seeds: int = 2
        n_trials: int = 4
        n_epochs: int = 3

        dev: qml.Device = qml.device("lightning.qubit", wires=n_qubits)

        self.binary_classifier_finder: ModelFinder = ModelFinder(
            task_type=MLTaskType.BINARY_CLASSIFICATION,
            features=x,
            classes=y,
            n_cores=1,
            n_trials=n_trials,
            n_seeds=n_seeds,
            n_epochs=n_epochs,
            device=dev,
        )

        self.classifier_finder: ModelFinder = ModelFinder(
            task_type=MLTaskType.CLASSIFICATION,
            features=cls_x,
            classes=cls_y,
            n_cores=1,
            n_trials=n_trials,
            n_seeds=n_seeds,
            n_epochs=n_epochs,
            device=dev,
        )

        self.linear_regressor_finder: ModelFinder = ModelFinder(
            task_type=MLTaskType.REGRESSION,
            features=reg_x,
            classes=reg_y,
            n_cores=1,
            n_trials=n_trials,
            n_seeds=n_seeds,
            n_epochs=n_epochs,
            device=dev,
        )

    def test_binary_classification_model_finding(self) -> None:
        """
        Tests if `ModelFinder` finds a binary classification model.
        """
        self.binary_classifier_finder.find_model()
        self.assertTrue(
            True, "ModelFinder crashed while finding a binary classification model!"
        )

    def test_classification_model_finding(self) -> None:
        """
        Tests if `ModelFinder` finds a linear regression model.
        """
        self.classifier_finder.find_model()
        self.assertTrue(
            True, "ModelFinder crashed while finding a classification model!"
        )

    def test_linear_regression_model_finding(self) -> None:
        """
        Tests if `ModelFinder` finds a linear regression model.
        """
        self.linear_regressor_finder.find_model()
        self.assertTrue(
            True, "ModelFinder crashed while finding a linear regression model!"
        )


class TestHyperparameterTuner(unittest.TestCase):
    """
    This is a `TestCase` for the `HyperparameterTuner` class.
    """

    def setUp(self) -> None:
        """
        Sets up the tests.

        :Note:
            There's no return value and only a single method to test, so we strive
            to be as minimalistic as possible.
        """
        x: Sequence[Sequence[float]]
        y: Sequence[int]

        x, y = make_moons(
            n_samples=100,
            shuffle=True,
            noise=0.1,
            random_state=RandomState(0),
        )

        n_seeds: int = 2
        n_trials: int = 2
        n_qubits: int = 2

        dev: qml.Device = qml.device("lightning.qubit", wires=n_qubits)

        classifier: QNNBinaryClassifier = QNNBinaryClassifier(2, 20, 5, device=dev)

        self.tuner: HyperparameterTuner = HyperparameterTuner(
            x,
            y,
            classifier,
            n_seeds=n_seeds,
            n_trials=n_trials,
        )

    def test_hyperparameter_tuner_running(self) -> None:
        """
        Tests if `HyperparameterTuner` runs.
        """
        self.tuner.find_hyperparameters()
        self.assertTrue(True, "HyperparameterTuner crashed during tuning!")
