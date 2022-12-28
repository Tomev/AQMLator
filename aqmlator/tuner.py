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

import uuid
import optuna
import pennylane

from optuna.samplers import TPESampler
from typing import Sequence, List, Dict, Any
from enum import IntEnum

from pennylane.templates.embeddings import AmplitudeEmbedding, AngleEmbedding

from aqmlator.qnn import QNNBinaryClassifier


class DataEmbedding(IntEnum):
    AMPLITUDE: int = 0
    ANGLE: int = 1


class ModelFinder:
    """
    A class for finding the best QNN model for given data.
    """

    def __init__(
        self,
        features: Sequence[Sequence[float]],
        classes: Sequence[int],
        study_name: str = "QML_Model_Finder_",
        add_uuid: bool = True,
        minimal_accuracy: float = 0.8,
        batch_size: int = 20,
        n_cores: int = -1,
        n_trials: int = 100,
        n_epochs: int = 10,
        n_seeds: int = 5,
    ):
        """
        A constructor for `ModelFinder` class.

        :param features:
            Features of the objects to be classified. Their order should correspond to
            that of `classes`.
        :param classes:
            Classes of the classified objects. Their order should correspond to that
            of `features`.
        :param study_name:
            The name of the `optuna` study. If the study with this name is already
            stored in the
        :param add_uuid:
            Flag for specifying if uuid should be added to the `study_name`.
        :param minimal_accuracy:
            Minimal accuracy after which the classification
        :param n_cores:
            The number of cores that `optuna` can use. The (default) value :math:`-1`
            means all of them.
        :param n_trials:
            The number of trials after which the search will be finished.
        :param n_epochs:
            The number of QNN training epochs.
        :param n_seeds:
            Number of seeds checked per `optuna` trial.
        """
        self._n_trials: int = n_trials
        self._n_cores: int = n_cores
        self._n_epochs: int = n_epochs
        self._n_seeds: int = n_seeds
        self._batch_size: int = batch_size
        self._minimal_accuracy: float = minimal_accuracy

        self._x: Sequence[Sequence[float]] = features
        self._y: Sequence[int] = classes

        self._study_name: str = study_name

        if add_uuid:
            self._study_name += str(uuid.uuid1())

        self._embeddings: List[pennylane.operation.Operation] = [
            AmplitudeEmbedding,
            AngleEmbedding,
        ]

    def find_model(self) -> None:
        """
        Finds the QNN model that best fits the given data.
        """
        sampler: TPESampler = TPESampler(
            seed=0, multivariate=True, group=True  # For experiments repeatability.
        )

        study: optuna.study.Study = optuna.create_study(
            sampler=sampler, study_name=self._study_name, load_if_exists=True
        )

        study.optimize(
            self._optuna_objective, n_trials=self._n_trials, n_jobs=self._n_cores
        )

    def _optuna_objective(self, trial: optuna.trial.Trial) -> float:
        """
        Objective function of the `optuna` optimizer.

        :Note:
            Instead of optimizing the hyperparameters, as `optuna` usually does, this
            optimizes the structure of the VQC.

        :param trial:
            The `optuna` Trial object used to randomize and store the

        :return:
            The average number of calls made to the quantum device (which `optuna`
            wants to minimize).
        """
        embedding_index: int = trial.suggest_int(
            "embedding", 0, len(self._embeddings) - 1
        )

        n_layers: int = trial.suggest_int("n_layers", 1, 3)
        n_qubits: int = len(self._x[0])

        embedding_kwargs: Dict[str, Any] = {}
        embedding_kwargs["wires"] = range(n_qubits)

        if embedding_index == DataEmbedding.AMPLITUDE:
            embedding_kwargs["pad_with"] = 0
            embedding_kwargs["normalize"] = True

        quantum_device_calls: int = 0

        for seed in range(self._n_seeds):
            classifier: QNNBinaryClassifier = QNNBinaryClassifier(
                n_qubits,
                n_layers,
                self._batch_size,
                n_epochs=self._n_epochs,
                embedding_method=self._embeddings[embedding_index],
                embedding_kwargs=embedding_kwargs,
                accuracy_threshold=self._minimal_accuracy,
                weights_random_seed=seed,
            )

            classifier.fit(self._x, self._y)

            quantum_device_calls += classifier.n_executions()

        return quantum_device_calls / self._n_seeds
