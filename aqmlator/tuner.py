"""
=============================================================================

    This module contains the classes that use optuna for different kinds of
    optimizations - mainly model and hyperparameter.

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

import abc
import uuid
import optuna
import pennylane

import pennylane.numpy as np

from optuna.samplers import TPESampler
from typing import Sequence, List, Dict, Any, Tuple, Type, Callable
from enum import IntEnum

from pennylane.templates.embeddings import AmplitudeEmbedding, AngleEmbedding
from pennylane.templates.layers import StronglyEntanglingLayers, BasicEntanglerLayers
from pennylane.optimize import (
    NesterovMomentumOptimizer,
    AdamOptimizer,
    GradientDescentOptimizer,
)

from aqmlator.qml import (
    QuantumKernelBinaryClassifier,
    QNNBinaryClassifier,
    QMLModel,
    QNNLinearRegression,
)


class MLTaskType(IntEnum):
    BINARY_CLASSIFICATION: int = 0
    CLASSIFICATION: int = 1
    REGRESSION: int = 2


class BinaryClassifierType(IntEnum):
    QNN: int = 0
    QEK: int = 1


class DataEmbedding(IntEnum):
    AMPLITUDE: int = 0
    ANGLE: int = 1


class Layers(IntEnum):
    BASIC: int = 0
    STRONGLY_ENTANGLING: int = 1


class Optimizers(IntEnum):
    NESTEROV: int = 0
    ADAM: int = 1


class OptunaOptimizer(abc.ABC):
    """
    A class for all `optuna`-based optimizers that takes care of the common boilerplate
    code, especially in the constructor.
    """

    def __init__(
        self,
        features: Sequence[Sequence[float]],
        classes: Sequence[int],
        study_name: str = "",
        add_uuid: bool = True,
        n_trials: int = 10,
        n_cores: int = 1,
        n_seeds: int = 1,
    ) -> None:
        """
        A constructor for the `OptunaOptimizer` class.

        :param features:
            The lists of features of the objects that are used during the training.
        :param classes:
            A list of classes or function values corresponding to the given lists of
            features.
        :param study_name:
            The name of the `optuna` study. If the study with this name is already
            stored in the DB, the tuner will continue the study.
        :param add_uuid:
            Flag for specifying if uuid should be added to the `study_name`.
        :param n_cores:
            The number of cores that `optuna` can use. The (default) value :math:`-1`
            means all of them.
        :param n_trials:
            The number of trials after which the search will be finished.
        :param n_seeds:
            Number of seeds checked per `optuna` trial.
        """
        self._x: Sequence[Sequence[float]] = features
        self._y: Sequence[int] = classes

        self._study_name: str = study_name

        if add_uuid:
            self._study_name += str(uuid.uuid1())

        self._n_trials: int = n_trials
        self._n_cores: int = n_cores
        self._n_seeds: int = n_seeds
        pass


class ModelFinder(OptunaOptimizer):
    """
    A class for finding the best QNN model for given data and task.
    """

    def __init__(
        self,
        task_type: int,
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
            stored in the DB, the finder will continue the study.
        :param add_uuid:
            Flag for specifying if uuid should be added to the `study_name`.
        :param minimal_accuracy:
            Minimal accuracy after which the training will end.
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
        super().__init__(
            features, classes, study_name, add_uuid, n_trials, n_cores, n_seeds
        )

        self._task_type: int = task_type

        self._n_epochs: int = n_epochs

        self._batch_size: int = batch_size
        self._minimal_accuracy: float = minimal_accuracy

        self._binary_classifiers: List[Type[QMLModel]] = [
            QNNBinaryClassifier,
            QuantumKernelBinaryClassifier,
        ]

        self._binary_classifiers_kwargs_generator: List[
            Callable[[optuna.trial.Trial], Dict[str, Any]]
        ] = [
            self._get_qnn_based_model_kwargs,
            self._get_qek_binary_classifier_kwargs,
        ]

        self._quantum_linear_regressors: List[Type[QMLModel]] = [QNNLinearRegression]

        self._linear_regressors_kwargs_generator: List[
            Callable[[optuna.trial.Trial], Dict[str, Any]]
        ] = [
            self._get_qnn_based_model_kwargs,
        ]

        self._embeddings: List[Type[pennylane.operation.Operation]] = [
            AmplitudeEmbedding,
            AngleEmbedding,
        ]

        self._layers: List[Type[pennylane.operation.Operation]] = [
            BasicEntanglerLayers,
            StronglyEntanglingLayers,
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

        optuna_objective: Callable[
            [optuna.trial.Trial], float
        ] = self._binary_classification_objective_function

        if self._task_type == MLTaskType.REGRESSION:
            optuna_objective = self._regression_objective_function

        study.optimize(optuna_objective, n_trials=self._n_trials, n_jobs=self._n_cores)

    def _binary_classification_objective_function(
        self, trial: optuna.trial.Trial
    ) -> float:
        """
        Objective function of the `optuna` optimizer for binary classification model
        finder.

        :Note:
            Instead of optimizing the hyperparameters, as `optuna` usually does, this
            optimizes the structure of the VQC for binary classification.

        :param trial:
            The `optuna` Trial object used to randomize and store the results of the
            optimization.

        :return:
            The average number of calls made to the quantum device (which `optuna`
            wants to minimize).
        """
        quantum_device_calls: int = 0

        classifier_type: int = trial.suggest_int(
            "classifier_type", 0, len(self._binary_classifiers) - 1
        )
        kwargs: Dict[str, Any] = self._binary_classifiers_kwargs_generator[
            classifier_type
        ](trial)

        kwargs["n_epochs"] = self._n_epochs
        kwargs["accuracy_threshold"] = self._minimal_accuracy
        kwargs["rng_seed"] = 0

        self._suggest_layers(trial, kwargs)

        for seed in range(self._n_seeds):

            kwargs["rng_seed"] = seed

            classifier: QMLModel = self._binary_classifiers[classifier_type](**kwargs)

            classifier.fit(self._x, self._y)

            quantum_device_calls += classifier.n_executions()

        return quantum_device_calls / self._n_seeds

    def _classification_objective_function(self, trial: optuna.trial.Trial) -> float:
        """

        :param trial:
        :return:
        """
        raise NotImplementedError

    def _regression_objective_function(self, trial: optuna.trial.Trial) -> float:
        """
        Objective function of the `optuna` optimizer for QNN regression model finder.

        :Note:
            Instead of optimizing the hyperparameters, as `optuna` usually does, this
            optimizes the structure of the VQC for binary classification.

        :param trial:
            The `optuna` Trial object used to randomize and store the results of the
            optimization.

        :return:
            The average number of calls made to the quantum device (which `optuna`
            wants to minimize).
        """
        quantum_device_calls: int = 0

        regressor_type: int = trial.suggest_int(
            "regressor_type", 0, len(self._quantum_linear_regressors) - 1
        )
        kwargs: Dict[str, Any] = self._linear_regressors_kwargs_generator[
            regressor_type
        ](trial)

        kwargs["n_epochs"] = self._n_epochs
        kwargs["accuracy_threshold"] = self._minimal_accuracy
        kwargs["rng_seed"] = 0

        self._suggest_layers(trial, kwargs)

        for seed in range(self._n_seeds):
            kwargs["rng_seed"] = seed

            regressor: QMLModel = self._quantum_linear_regressors[regressor_type](
                **kwargs
            )

            regressor.fit(self._x, self._y)

            quantum_device_calls += regressor.n_executions()

        return quantum_device_calls / self._n_seeds

    def _suggest_layers(
        self, trial: optuna.trial.Trial, kwargs: Dict[str, Any]
    ) -> None:
        """
        Using `optuna`, suggest the order of layers in the VQC based on the `kwargs`
        given.

        :param trial:
            Optuna `Trial` object that "suggests" the parameters values.
        :param kwargs:
            A dictionary of keyword arguments that will be used to initialize the
            QML model.
        """
        layers: List[Type[pennylane.operation.Operation]] = []
        layers_weights_shapes: List[Tuple[int, ...]] = []

        for i in range(kwargs["n_layers"]):
            layer_index: int = trial.suggest_int(f"layer_{i}", 0, len(self._layers) - 1)
            layers.append(self._layers[layer_index])

            if layer_index == Layers.BASIC:
                layers_weights_shapes.append((1, kwargs["wires"]))

            if layer_index == Layers.STRONGLY_ENTANGLING:
                layers_weights_shapes.append((1, kwargs["wires"], 3))

        kwargs["layers"] = layers
        kwargs["layers_weights_shapes"] = layers_weights_shapes
        kwargs.pop("n_layers")

    def _get_qnn_based_model_kwargs(self, trial: optuna.trial.Trial) -> Dict[str, Any]:
        """
        Prepares the dict of kwargs for `QNNBinaryClassifier` class.

        :param trial:
            Optuna `Trial` object that "suggests" the parameters values.

        :return:
            A dictionary with fields required for proper `QNNBinaryClassifier`
            construction.
        """
        kwargs: Dict[str, Any] = {"wires": len(self._x[0])}

        embedding_index: int = trial.suggest_int(
            "embedding", 0, len(self._embeddings) - 1
        )

        kwargs["embedding_method"] = self._embeddings[embedding_index]

        embedding_kwargs: Dict[str, Any] = {"wires": range(kwargs["wires"])}

        if embedding_index == DataEmbedding.AMPLITUDE:
            embedding_kwargs["pad_with"] = 0
            embedding_kwargs["normalize"] = True

        kwargs["embedding_kwargs"] = embedding_kwargs

        kwargs["n_layers"] = trial.suggest_int("n_layers", 1, 3)
        kwargs["batch_size"] = self._batch_size

        return kwargs

    def _get_qek_binary_classifier_kwargs(
        self, trial: optuna.trial.Trial
    ) -> Dict[str, Any]:
        """
        Prepares the dict of kwargs for `QuantumKernelBinaryClassifier` class.

        :param trial:
            Optuna `Trial` object that "suggests" the parameters values.

        :return:
            A dictionary with fields required for proper
            `QuantumKernelBinaryClassifier` construction.
        """
        kwargs: Dict[str, Any] = {
            "wires": len(self._x[0]),
            "n_layers": trial.suggest_int("n_layers", 3, 5),
        }

        return kwargs


class HyperparameterTuner(OptunaOptimizer):
    """
    This class contains the optuna-based tuner for ML Training hyperparameters.
    """

    def __init__(
        self,
        features: Sequence[Sequence[float]],
        classes: Sequence[int],
        model: QMLModel,
        study_name: str = "QML_Hyperparameter_Tuner_",
        add_uuid: bool = True,
        n_trials: int = 10,
        n_cores: int = 1,
        n_seeds: int = 1,
    ) -> None:
        """
        A constructor for the `HyperparameterTuner` class.

        :param features:
            The lists of features of the objects that are used during the training.
        :param classes:
            A list of classes corresponding to the given lists of features.
        :param model:
            A model to be trained.
        :param study_name:
            The name of the `optuna` study. If the study with this name is already
            stored in the DB, the tuner will continue the study.
        :param add_uuid:
            Flag for specifying if uuid should be added to the `study_name`.
        :param n_cores:
            The number of cores that `optuna` can use. The (default) value :math:`-1`
            means all of them.
        :param n_trials:
            The number of trials after which the search will be finished.
        :param n_seeds:
            Number of seeds checked per `optuna` trial.
        """
        super().__init__(
            features, classes, study_name, add_uuid, n_trials, n_cores, n_seeds
        )

        self._optimizers: List[Callable[[Any], GradientDescentOptimizer]] = [
            AdamOptimizer,
            NesterovMomentumOptimizer,
        ]

        self._model: QMLModel = model

    def find_hyperparameters(self) -> None:
        """
        Finds the (sub)optimal training hyperparameters.
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

    def _suggest_optimizer(self, trial: optuna.trial.Trial) -> GradientDescentOptimizer:
        """

        :param trial:
            The `optuna.trial.Trial` object used to randomize and store the results of
            the optimization.
        :return:
            The suggested optimizer.
        """
        optimizer_index: int = trial.suggest_int(
            "optimizer", 0, len(self._optimizers) - 1
        )

        optimizer: GradientDescentOptimizer

        if optimizer_index == Optimizers.ADAM:
            # Hyperparameters range taken from arXiv:1412.6980.
            optimizer = AdamOptimizer(
                stepsize=trial.suggest_float("stepsize", 0.00001, 0.1),
                beta1=trial.suggest_float("beta1", 0, 0.9),
                beta2=trial.suggest_float("beta2", 0.99, 0.9999),
            )

        if optimizer_index == Optimizers.NESTEROV:
            # Hyperparameters range taken from
            # https://cs231n.github.io/neural-networks-3/
            optimizer = NesterovMomentumOptimizer(
                stepsize=trial.suggest_float("stepsize", 0.00001, 0.1),
                momentum=trial.suggest_float("momentum", 0.5, 0.9),
            )

        return optimizer

    def _optuna_objective(self, trial: optuna.trial.Trial) -> float:
        """
        Objective function of the `optuna` optimizer.

        :param trial:
            The `optuna` Trial object used to randomize and store the results of the
            optimization.
        :return:
        """
        self._model.optimizer = self._suggest_optimizer(trial)

        initial_executions: int
        executions_sum: int = 0

        for _ in range(self._n_seeds):
            initial_executions = self._model.n_executions()
            self._model.weights = np.zeros_like(self._model.weights)
            self._model.fit(self._x, self._y)
            executions_sum += self._model.n_executions() - initial_executions

        return executions_sum / self._n_seeds
