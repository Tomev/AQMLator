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
from typing import Sequence, List, Dict, Any, Tuple, Type, Callable, Optional
from enum import StrEnum, auto

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
    QNNClassifier,
)

import aqmlator.database_connection as db

binary_classifiers: Dict[str, Dict[str, Any]] = {
    "QNN": {
        "constructor": QNNBinaryClassifier,
        "kwargs": {
            "batch_size": (15, 25),  # Might need to be data size-dependent instead.
        },
        "fixed_kwargs": {},
        "n_layers": (1, 3),
    },
    "QEK": {
        "constructor": QuantumKernelBinaryClassifier,
        "kwargs": {},
        "fixed_kwargs": {},
        "n_layers": (3, 5),
    },
}

regressors: Dict[str, Dict[str, Any]] = {
    "QNN": {
        "kwargs": {
            "batch_size": (15, 25),  # Might need to be data size-dependent instead.
        },
        "n_layers": (1, 3),
        "constructor": QNNLinearRegression,
    }
}

data_embeddings: Dict[str, Dict[str, Any]] = {
    "ANGLE": {"constructor": AngleEmbedding, "kwargs": {}, "fixed_kwargs": {}},
    "AMPLITUDE": {
        "constructor": AmplitudeEmbedding,
        "kwargs": {},
        "fixed_kwargs": {"pad_with": 0, "normalize": True},
    },
}

optimizers: Dict[str, Dict[str, Any]] = {
    "NESTEROV": {
        "constructor": NesterovMomentumOptimizer,
        # Hyperparameters range taken from
        # https://cs231n.github.io/neural-networks-3/
        "kwargs": {
            "stepsize": {"min": 0.00001, "max": 0.1},
            "momentum": {"min": 0.5, "max": 0.9},
        },
    },
    "ADAM": {
        "constructor": AdamOptimizer,
        # Hyperparameters range taken from arXiv:1412.6980.
        "kwargs": {
            "stepsize": {
                "min": 0.00001,
                "max": 0.1,
            },
            "beta1": {
                "min": 0,
                "max": 0.9,
            },
            "beta2": {"min": 0.99, "max": 0.9999},
        },
    },
}


class MLTaskType(StrEnum):
    BINARY_CLASSIFICATION: str = auto()
    CLASSIFICATION: str = auto()
    REGRESSION: str = auto()


class Layers(StrEnum):
    BASIC: str = "BASIC"
    STRONGLY_ENTANGLING: str = "STRONGLY_ENTANGLING"


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

    @staticmethod
    def _get_storage() -> Optional[str]:
        return db.get_database_url()


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

        # A dict that will be string models taken under consideration during the
        # model finding.
        self._models_dict: Dict[str, Any] = {}

        self._task_type: int = task_type

        self._n_epochs: int = n_epochs

        self._minimal_accuracy: float = minimal_accuracy

        self._optuna_objective_functions: Dict[
            str, Callable[[optuna.trial.Trial], float]
        ] = {
            MLTaskType.BINARY_CLASSIFICATION: self._simple_model_objective_function,
            MLTaskType.CLASSIFICATION: self._classification_objective_function,
            MLTaskType.REGRESSION: self._simple_model_objective_function,
        }

        self._layers: Dict[Layers, Type[pennylane.operation.Operation]] = {
            Layers.BASIC: BasicEntanglerLayers,
            Layers.STRONGLY_ENTANGLING: StronglyEntanglingLayers,
        }

        self._optuna_postfix: str = ""

    def find_model(self) -> None:
        """
        Finds the QNN model that best fits the given data.
        """
        sampler: TPESampler = TPESampler(
            seed=0, multivariate=True, group=True  # For experiments repeatability.
        )

        study: optuna.study.Study = optuna.create_study(
            sampler=sampler,
            study_name=self._study_name,
            load_if_exists=True,
            storage=self._get_storage(),
        )

        study.optimize(
            self._optuna_objective_functions[self._task_type],
            n_trials=self._n_trials,
            n_jobs=self._n_cores,
        )

    def _simple_model_objective_function(self, trial: optuna.trial.Trial) -> float:
        """
        Default objective function of the `optuna` optimizer for the model finding. It
        is meant to work for all the simple (single) models.

        :Note:
            Instead of optimizing the hyperparameters, as `optuna` usually does, this
            optimizes the structure of the VQC for the model.

        :param trial:
            The `optuna` Trial object used to randomize and store the results of the
            optimization.

        :return:
            The average number of calls made to the quantum device (which `optuna`
            wants to minimize).
        """
        self._initialize_model_dict()

        model_type: str = trial.suggest_categorical(
            "model_type" + self._optuna_postfix, list(self._models_dict.keys())
        )

        kwargs: Dict[str, Any] = self._suggest_model_kwargs(trial, model_type)

        model: QMLModel = self._models_dict[model_type]["constructor"](**kwargs)

        return self._evaluate_model(model)

    def _initialize_model_dict(self) -> None:
        """
        Initializes the models dict used during

        :Note:
            The (non-binary) classification task uses binary classification dict to
            create a complex (classification) model.
        """
        self._models_dict = binary_classifiers

        if self._task_type == MLTaskType.REGRESSION:
            self._models_dict = regressors

    def _evaluate_model(self, model: QMLModel) -> float:
        """
        Evaluates the performance of the given model. The evaluation is based on the
        number of calls to the quantum machine (which are _expensive_) during the
        fitting.

        :param model:
            A `QMLModel` to evaluate.

        The mean number of calls to the quantum machine during the fitting.
        """
        quantum_device_calls: int = 0

        for seed in range(self._n_seeds):
            model.seed(seed)

            model.fit(self._x, self._y)

            quantum_device_calls += model.n_executions()

        return quantum_device_calls / self._n_seeds

    def _classification_objective_function(self, trial: optuna.trial.Trial) -> float:
        """
        Objective function of the `optuna` optimizer for classification model finder.

        :param trial:
            The `optuna` Trial object used to randomize and store the results of the
            optimization.

        :return:
            The average number of calls made to the quantum device (which `optuna`
            wants to minimize).
        """
        self._initialize_model_dict()

        quantum_device_calls: int = 0

        n_classes: int = len(np.unique(self._y))

        binary_classifiers_kwargs: List[Dict[str, Any]] = []

        for i in range(n_classes):
            self._optuna_postfix = f"_({i})"

            kwargs: Dict[str, Any] = self._suggest_model_kwargs(trial, "QNN")

            binary_classifiers_kwargs.append(kwargs)

        qnn_binary_classifiers: List[QNNBinaryClassifier] = []

        for i in range(n_classes):
            qnn_binary_classifiers.append(
                QNNBinaryClassifier(**binary_classifiers_kwargs[i])
            )

        classifier: QNNClassifier = QNNClassifier(
            wires=range(len(self._x)),
            n_classes=n_classes,
            binary_classifiers=qnn_binary_classifiers,
        )

        self._optuna_postfix = ""

        return self._evaluate_model(classifier)

    def _suggest_model_kwargs(
        self, trial: optuna.trial.Trial, model_type: str
    ) -> Dict[str, Any]:
        """
        Suggests the kwargs for the specified (qml) model.

        :param trial:
            An optuna trial object for parameters selection.
        :param model_type:
            A string describing the type of the model. Note that the
            `self._models_dict` has to be initialized properly, prior to calling
            this method.

        :return:
            Returns the suggested kwargs for the selected (qml) model type.
        """
        kwargs: Dict[str, Any] = {
            "wires": len(self._x[0]),
            "n_layers": trial.suggest_int(
                "n_layers" + self._optuna_postfix,
                self._models_dict[model_type]["n_layers"][0],
                self._models_dict[model_type]["n_layers"][1],
            ),
            "n_epochs": self._n_epochs,
            "accuracy_threshold": self._minimal_accuracy,
        }

        # TR: Might need to be extended for different arguments type at some point.
        kwargs_data: Dict[str, any] = self._models_dict[model_type]["kwargs"]

        for kwarg in kwargs_data:
            kwargs[kwarg] = trial.suggest_int(
                kwarg + self._optuna_postfix,
                kwargs_data[kwarg][0],
                kwargs_data[kwarg][1],
            )

        self._suggest_embedding(trial, kwargs)
        self._suggest_layers(trial, kwargs)

        return kwargs

    def _suggest_embedding(
        self, trial: optuna.trial.Trial, kwargs: Dict[str, Any]
    ) -> None:
        """
        Using 'optuna', suggest the embedding and its `kwargs`. Everything is then added
        to the given `kwargs`.

        :param trial:
            Optuna `Trial` object that "suggests" the parameters values.
        :param kwargs:
            A dictionary of keyword arguments that will be used to initialize the
            QML model.
        """
        embedding_type: str = trial.suggest_categorical(
            "embedding" + self._optuna_postfix, [e for e in data_embeddings]
        )

        kwargs["embedding_method"] = data_embeddings[embedding_type]["constructor"]

        embedding_kwargs: Dict[str, Any] = {"wires": range(kwargs["wires"])}

        embedding_kwargs.update(data_embeddings[embedding_type]["fixed_kwargs"])

        kwargs["embedding_kwargs"] = embedding_kwargs

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
            layer_type: int = trial.suggest_categorical(
                f"layer_{i}" + self._optuna_postfix, [l for l in Layers]
            )
            layers.append(self._layers[layer_type])

            if layer_type == Layers.BASIC:
                layers_weights_shapes.append((1, kwargs["wires"]))

            if layer_type == Layers.STRONGLY_ENTANGLING:
                layers_weights_shapes.append((1, kwargs["wires"], 3))

        kwargs["layers"] = layers
        kwargs["layers_weights_shapes"] = layers_weights_shapes
        kwargs.pop("n_layers")


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

        self._model: QMLModel = model

    def find_hyperparameters(self) -> None:
        """
        Finds the (sub)optimal training hyperparameters.
        """
        sampler: TPESampler = TPESampler(
            seed=0, multivariate=True, group=True  # For experiments repeatability.
        )

        study: optuna.study.Study = optuna.create_study(
            sampler=sampler,
            study_name=self._study_name,
            load_if_exists=True,
            storage=self._get_storage(),
        )

        study.optimize(
            self._optuna_objective, n_trials=self._n_trials, n_jobs=self._n_cores
        )

    @staticmethod
    def _suggest_optimizer(trial: optuna.trial.Trial) -> GradientDescentOptimizer:
        """

        :param trial:
            The `optuna.trial.Trial` object used to randomize and store the results of
            the optimization.
        :return:
            The suggested optimizer.
        """

        optimizer_type: str = trial.suggest_categorical(
            "optimizer", [o for o in optimizers]
        )

        kwargs_data: Dict[str, any] = optimizers[optimizer_type]["kwargs"]
        kwargs: Dict[str, any] = dict()

        # TR: Might need rebuilding for int and str kwargs.
        for kwarg in kwargs_data:
            kwargs[kwarg] = trial.suggest_float(
                kwarg, kwargs_data[kwarg]["min"], kwargs_data[kwarg]["max"]
            )

        optimizer: GradientDescentOptimizer = optimizers[optimizer_type]["constructor"](
            **kwargs
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
