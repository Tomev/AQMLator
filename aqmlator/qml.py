"""
=============================================================================

    This module contains the functionalities related to the quantum machine learning.

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
import random
from itertools import chain
from math import prod
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import pennylane as qml
import torch
from dimod.core.sampler import Sampler
from lightning.pytorch.trainer import Trainer
from numpy.typing import NDArray
from pennylane import numpy as np
from pennylane.kernels import target_alignment
from pennylane.optimize import GradientDescentOptimizer, NesterovMomentumOptimizer
from pennylane.templates.embeddings import AmplitudeEmbedding, AngleEmbedding
from pennylane.templates.layers import StronglyEntanglingLayers
from pennylane.transforms import transpile
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from torch import Tensor
from torch.utils.data import DataLoader

from qbm4eo.encoder import LBAEEncoder
from qbm4eo.lbae import LBAE
from qbm4eo.rbm import RBM, AnnealingRBMTrainer, CD1Trainer, RBMTrainer

ModelOutput = TypeVar("ModelOutput", float, int)


class QMLModel(abc.ABC):
    """
    A boilerplate class, providing an interface for future QML models.
    """

    def __init__(
        self,
        wires: Union[int, Sequence[int]],
        device: Optional[qml.Device] = None,
        optimizer: Optional[GradientDescentOptimizer] = None,
        embedding_method: Optional[Type[qml.operation.Operation]] = None,
        embedding_kwargs: Optional[Dict[str, Any]] = None,
        layers: Optional[Sequence[Type[qml.operation.Operation]]] = None,
        validation_set_size: float = 0.2,
        rng_seed: int = 42,
        coupling_map: Optional[Sequence[Sequence[int]]] = None,
        n_qubit: Optional[int] = None,
    ) -> None:
        """
        The constructor for the `QMLModel` class.

        :param wires:
            The wires to use in the VQC or the number of qubits (and wires) used in the
            VQC.
        :param device:
            A device on which the model should operate.
        :param optimizer:
            The optimizer that will be used in the training. `NesterovMomentumOptimizer`
            with default parameters will be used as default.
        :param embedding_method:
            Embedding method of the data. By default - when `None` is specified - the
            QMLModel will use `AngleEmbedding`. See `_prepare_default_embedding`
            for parameters details.
        :param embedding_kwargs:
            Keyword arguments for the embedding method. If `None` are specified the
            QMLModel will use the default embedding method.
        :param layers:
            Layers to be used in the VQC. The layers will be applied in the given order.
            A double `StronglyEntanglingLayer` will be used if `None` is given.
        :param validation_set_size:
            A part of the training set that will be used for QMLModel validation.
            It should be from (0, 1).
        :param rng_seed:
            A seed used for random weights initialization.
        :param coupling_map:
            A description of connections between the qubits in the device.
        """
        self.dev: qml.Device = device

        self.wires: Sequence[int]

        if isinstance(wires, int):
            self.wires = list(range(wires))
        else:
            self.wires = wires

        self._training_X: Sequence[Sequence[float]]
        self._training_y: Sequence[int]
        self._validation_X: Sequence[Sequence[float]]
        self._validation_y: Sequence[int]

        self._embedding_method: Type[qml.operation.Operation]
        self._embedding_kwargs: Dict[str, Any]

        if embedding_method is None or embedding_kwargs is None:
            self._prepare_default_embedding()
        else:
            self._embedding_method = embedding_method
            self._embedding_kwargs = embedding_kwargs

        self._layers: Sequence[Type[qml.operation.Operation]]

        if layers is None:
            self._prepare_default_layers()
        else:
            self._layers = layers

        self._rng_seed: int = rng_seed
        self._rng: random.Random = random.Random(rng_seed)

        self._validation_set_size: float = validation_set_size

        if optimizer is None:
            optimizer = NesterovMomentumOptimizer()

        self.optimizer: GradientDescentOptimizer = optimizer
        self.weights: Sequence[float]

        self.coupling_map: Optional[Sequence[Sequence[int]]] = coupling_map

        if not n_qubit:
            n_qubit = len(self.wires)

        self.n_qubit: int = n_qubit

    def seed(self, new_seed: int) -> None:
        """
        Sets up the new seed.

        :param new_seed:
            New seed to be applied to the model.
        """
        self._rng_seed = new_seed

    def n_executions(self) -> int:
        """
        Returns number of VQC executions so far.

        :return:
            Returns the number of times the quantum device was called.
        """
        return self.dev.num_executions

    @abc.abstractmethod
    def fit(
        self,
        X: Union[Sequence[Sequence[float]], NDArray[np.float64]],
        y: Optional[Sequence[ModelOutput]],
    ) -> "QMLModel":
        """
        The model training method.

        :param X:
            The lists of features of the objects that are used during the training.
        :param y:
            A list of classes corresponding to the given lists of features.

        :return:
            Returns self after training.
        """
        raise NotImplementedError

    def _prepare_default_embedding(self) -> None:
        """
        Prepares the default embedding method is `None` was specified or if the
        kwargs were `None`. The default one is simple `AngleEmbedding`.
        """
        self._embedding_method = AngleEmbedding
        self._embedding_kwargs = {"wires": self.wires}

    def _prepare_default_layers(self) -> None:
        """
        Prepares the default layers of the model if `None` was given in either
        `layers` or `layers_weights_number` arguments of the constructor. We will
        use a double strongly entangling layer.
        """
        self._layers = [StronglyEntanglingLayers] * 2

    def _split_data_for_training(
        self,
        X: Union[Sequence[Sequence[float]], NDArray[np.float64]],
        y: Sequence[ModelOutput],
    ) -> None:
        """
        Splits the objects into validation and training sets. Should be called before
        the training, usually near the beginning of the `fit` method.

        :param X:
            All object features that will be split into training and validation set.
        :param y:
            Corresponding classes that will be split into training and validation set.
        """
        (
            self._training_X,
            self._validation_X,
            self._training_y,
            self._validation_y,
        ) = train_test_split(
            X,
            y,
            test_size=self._validation_set_size,
            random_state=self._rng_seed,
        )


class QNNModel(QMLModel, abc.ABC):
    """
    A boilerplate class, providing an interface for future QNN-Based models.
    """

    def __init__(
        self,
        wires: Union[int, Sequence[int]],
        batch_size: int,
        n_epochs: int = 1,
        device: Optional[qml.Device] = None,
        optimizer: Optional[GradientDescentOptimizer] = None,
        embedding_method: Optional[Type[qml.operation.Operation]] = None,
        embedding_kwargs: Optional[Dict[str, Any]] = None,
        layers: Optional[Sequence[Type[qml.operation.Operation]]] = None,
        accuracy_threshold: float = 0.8,
        initial_weights: Optional[Sequence[float]] = None,
        rng_seed: int = 42,
        validation_set_size: float = 0.2,
        prediction_function: Optional[
            Callable[
                [Sequence[Sequence[float]]],
                Sequence[ModelOutput],
            ]
        ] = None,
        debug_flag: bool = True,
        coupling_map: Optional[Sequence[Sequence[int]]] = None,
        n_qubit: Optional[int] = None,
    ) -> None:
        """
        The constructor for the `QNNModel` class.

        :param wires:
            The wires to use in the VQC or the number of qubits (and wires) used in the
            VQC.
        :param batch_size:
            Size of a batches used during the training.
        :param n_epochs:
            The number of training epochs.
        :param device:
            A device on which the model should operate.
        :param optimizer:
            The optimizer that will be used in the training. `NesterovMomentumOptimizer`
            with default parameters will be used as default.
        :param embedding_method:
            Embedding method of the data. By default - when `None` is specified - the
            model will use `AmplitudeEmbedding`. See `_prepare_default_embedding`
            for parameters details.
        :param embedding_kwargs:
            Keyword arguments for the embedding method. If none are specified the
            model will use the default embedding method.
        :param layers:
            Layers to be used in the VQC. The layers will be applied in the given order.
            A double `StronglyEntanglingLayer` will be used if `None` is given.
        :param accuracy_threshold:
            The satisfactory accuracy of the model.
        :param initial_weights:
            The initial weights for the training.
        :param rng_seed:
            A seed used for random weights initialization.
        :param validation_set_size:
            A part of the training set that will be used for model validation.
            It should be from (0, 1).
        :param prediction_function:
            A prediction function that will be used to process the output of the VQC.
            If `None` then the default one (for given model) will be used.
        :param debug_flag:
            A flag informing the model if the training info should be printed to the
            console or not.
        :param coupling_map:
            A description of connections between the qubits in the device.
        """

        super().__init__(
            wires,
            device,
            optimizer,
            embedding_method,
            embedding_kwargs,
            layers,
            validation_set_size,
            rng_seed,
            coupling_map,
            n_qubit,
        )

        self._n_epochs: int = n_epochs
        self._batch_size: int = batch_size

        self.accuracy_threshold: float = accuracy_threshold

        self._weights_length: int = 0

        for layer in self._layers:
            self._weights_length += prod(
                layer.shape(n_layers=1, n_wires=len(self.wires))
            )

        if initial_weights is None or len(initial_weights) != self._weights_length:
            initial_weights = [
                np.pi * self._rng.random() for _ in range(self._weights_length)
            ]

        self.weights = initial_weights

        self.circuit: Optional[qml.QNode] = None

        self._debug_flag: bool = debug_flag

        if prediction_function is None:
            prediction_function = self._default_prediction_function

        self._prediction_function: Callable[
            [Sequence[Sequence[float]]], Union[Sequence[int], Sequence[float]]
        ] = prediction_function

    def _create_circuit(self, interface: str = "autograd") -> qml.QNode:
        def circuit(
            inputs: Union[Sequence[float], torch.Tensor],
            weights: Union[np.ndarray, torch.Tensor],
        ) -> Sequence[float]:
            """
            Returns the expectation value of the first qubit of the VQC of which the
            weights are optimized during the learning process.

            :param inputs:
                Feature vector representing the object for which value is being
                predicted.

                :note:
                This argument needs to be named `inputs` for torch to be able to use
                the `circuit` method.
            :param weights:
                Weights that will be optimized during the learning process.

            :return:
                The expectation value (from range [-1, 1]) of the measurement in the
                computational basis of given circuit.
            """

            if isinstance(inputs, torch.Tensor):
                inputs = self._prepare_torch_inputs(inputs)

            for op in self._embedding_method(
                inputs, do_queue=False, **self._embedding_kwargs
            ).expand():
                qml.apply(op)

            start_weights: int = 0

            for i, layer in enumerate(self._layers):
                layer_shape: Tuple[int, ...] = layer.shape(
                    n_layers=1, n_wires=len(self.wires)
                )

                layer_weights = weights[
                    start_weights : start_weights + prod(layer_shape)
                ]

                start_weights += prod(layer_shape)

                layer_weights = layer_weights.reshape(layer_shape)

                for op in layer(
                    layer_weights, do_queue=False, wires=self.wires
                ).expand():
                    qml.apply(op)

            return [qml.expval(qml.PauliZ((i))) for i in self.wires]

        if self.coupling_map:
            circuit = transpile(coupling_map=self.coupling_map)(circuit)

        return qml.QNode(circuit, self.dev, interface=interface)

    def get_circuit_expectation_values(
        self, features_lists: Sequence[Sequence[float]]
    ) -> np.ndarray:
        """
        Computes and returns the expectation value of the `PauliZ` measurement of the
        first qubit of the VQC.

        :param features_lists:
            Features that will be encoded at the input of the circuit.

        :return:
            The expectation value of the `PauliZ` measurement on the first qubit of the
            VQC.
        """

        if not self.circuit:
            self.circuit = self._create_circuit()

        expectation_values: List[Sequence[float]] = []

        for features in features_lists:
            expectation_values.append(self.circuit(features, np.array(self.weights)))

        return np.array(expectation_values)

    @abc.abstractmethod
    def _cost(
        self,
        weights: Sequence[float],
        X: Sequence[Sequence[float]],
        y: Sequence[int],
    ) -> float:
        """
        Evaluates and returns the cost function value for the training purposes.

        :param X:
            The lists of features of the objects for which values are being predicted
            during the training.

        :param y:
            Outputs corresponding to the given features.

        :return:
            The value of the square loss function.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _default_prediction_function(
        self, circuit_outputs: Sequence[Sequence[float]]
    ) -> Sequence[ModelOutput]:
        """
        The default prediction function that should be specified for every QNN-based
        model.

        :param circuit_outputs:
            The output of the VQC.

        :return:
            Returns prediction value for the given problem.
        """
        pass

    def fit(
        self,
        X: Union[Sequence[Sequence[float]], NDArray[np.float64]],
        y: Optional[Sequence[ModelOutput]],
    ) -> "QNNModel":
        """
        The model training method.

        TODO TR: How to break this method down into smaller ones?

        :param X:
            The lists of features of the objects that are used during the training.
        :param y:
            A list of outputs corresponding to the given lists of features.

        :return:
            Returns `self` after training.
        """

        if not self.dev:
            raise AttributeError("Specify the device (dev) before fitting.")

        if y is None:
            raise AttributeError("Missing y in supervised learning model.")

        self.circuit = self._create_circuit()

        self._split_data_for_training(X, y)

        n_batches: int = max(1, len(X) // self._batch_size)

        feature_batches = np.array_split(np.arange(len(self._training_X)), n_batches)

        best_weights: Sequence[float] = self.weights
        best_accuracy: float = self.score(self._validation_X, self._validation_y)

        self.weights = np.array(self.weights, requires_grad=True)
        cost: float
        batch_indices: np.tensor  # Of ints.

        def _batch_cost(weights: Sequence[float]) -> float:
            """
            The cost function evaluated on the training data batch.

            :param weights:
                The weights to be applied to the VQC.

            :return:
                The value of `self._cost` function evaluated of the training data
                batch.
            """
            return self._cost(
                weights,
                self._training_X[batch_indices],
                self._training_y[batch_indices],
            )

        for it, batch_indices in enumerate(
            chain(*(self._n_epochs * [feature_batches]))
        ):
            # Update the weights by one optimizer step
            self.weights, cost = self.optimizer.step_and_cost(_batch_cost, self.weights)

            # Compute accuracy on the validation set
            accuracy_validation = self.score(self._validation_X, self._validation_y)

            # Make decision about stopping the training basing on the validation score
            if accuracy_validation >= best_accuracy:
                best_accuracy = accuracy_validation
                best_weights = self.weights

            if self._debug_flag:
                print(
                    f"It: {it + 1} / {self._n_epochs * n_batches} | Cost: {cost} |"
                    f" Accuracy (validation): {accuracy_validation}"
                )

            if accuracy_validation >= self.accuracy_threshold:
                break

        self.weights = best_weights

        # TR: Required for proper serialization.
        # TODO TR:  Maybe there's a better way to do it
        self.circuit = None

        return self

    @abc.abstractmethod
    def score(
        self,
        X: Sequence[Sequence[float]],
        y: Sequence[Union[int, float]],
        sample_weight: Sequence[float] = None,
    ) -> float:
        """
        Computes and returns score of the model.

        :note:
            There are no typehints, same as in sklearn `score` functions that we will
            use.

        :param X:
            Test samples.
        :param y:
            True labels for `X`.
        :param sample_weight:
            Sample weights.

        :return:
            Mean accuracy of `self.predict(X)` w.r.t. `y`.
        """
        raise NotImplementedError

    def _prepare_torch_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Torch inputs might need some manual preprocessing in some cases. This method
        handles this preprocessing.

        :param inputs:
            Inputs given by the
        :return:
            Preprocessed inputs.
        """
        # TODO TR: Think of better way to do it.
        if self._embedding_method == AmplitudeEmbedding:
            padding: torch.Tensor = torch.zeros([2 ** len(self.wires) - len(inputs)])
            inputs = torch.cat((inputs, padding))

        return inputs

    def get_torch_layer(self) -> torch.nn.Module:
        """
        This method creates a PyTorch (quantum) layer based on the VQC.

        :return:
            Returns a PyTorch Layer made from the VQC.
        """

        weight_shapes: Dict[str, int] = {"weights": len(self.weights)}
        return qml.qnn.TorchLayer(self._create_circuit("torch"), weight_shapes)

    def predict(
        self, features: Sequence[Sequence[float]]
    ) -> Union[Sequence[float], Sequence[int]]:
        """
        Returns predictions of the model for the given features.

        :param features:
            Features of the objects for which the model will predict the values.
        :return:
            Values predicted for given features.
        """

        self.circuit = self._create_circuit()

        results: Union[Sequence[int], Sequence[float]] = self._prediction_function(
            self.get_circuit_expectation_values(features)
        )

        # TR: Required for proper serialization.
        # TODO TR:  Maybe there's a better way to do it
        self.circuit = None

        return results


class QNNBinaryClassifier(ClassifierMixin, QNNModel):
    """
    This class implements a binary classifier that uses Quantum Neural Networks.

    The classifier expects two classes {0, 1}.
    """

    def _cost(
        self,
        weights: Sequence[float],
        X: Sequence[Sequence[float]],
        y: Sequence[int],
    ) -> float:
        """
        Evaluates and returns the cost function value for the training purposes.

        :param X:
            The lists of features of the objects that are being classified during the
            training.

        :param y:
            Classes corresponding to the given features.

        :return:
            The value of the square loss function.
        """

        self.circuit = self._create_circuit()

        expectation_values: np.ndarray = np.array(
            [self.circuit(x, weights)[0] for x in X]
        )

        return np.mean((expectation_values - np.array(y)) ** 2)

    def _default_prediction_function(
        self, circuit_outputs: Sequence[Sequence[float]]
    ) -> Sequence[ModelOutput]:
        """
        The default prediction function of the QNNClassifier.

        :param circuit_outputs:
            The outputs of the VQC.

        :return:
            Returns classification prediction value for the given problem.
        """
        return [2 * int(val >= 0.0) - 1 for val in [x[0] for x in circuit_outputs]]


class QNNLinearRegression(RegressorMixin, QNNModel):
    """
    This class implements a linear regressor that uses Quantum Neural Networks.
    """

    def _cost(
        self,
        weights: Sequence[float],
        X: Union[Sequence[Sequence[float]], NDArray[np.float64]],
        y: Sequence[int],
    ) -> float:
        """
        Evaluates and returns the cost function value for the training purposes.

        :param X:
            The lists of features of the objects that are being classified during the
            training.

        :param y:
            Classes corresponding to the given features.

        :return:
            The value of the square loss function.
        """
        self.circuit = self._create_circuit()

        expected_values = [self.circuit(x, weights) for x in X]

        predicted_values: np.ndarray = np.array(
            [
                sum(np.log(((i + 1) / 2) / (1 - ((i + 1) / 2))) for i in x)
                for x in expected_values
            ]
        )

        return np.mean(predicted_values - np.array(y) ** 2)

    def _default_prediction_function(
        self, circuit_outputs: Sequence[Sequence[float]]
    ) -> Sequence[ModelOutput]:
        """
        The default prediction function of the QNNClassifier.

        :param circuit_outputs:
            The outputs of the VQC.

        :return:
            Returns classification prediction value for the given problem.
        """
        predicted_values: List[ModelOutput] = [
            sum(np.log(((i + 1) / 2) / (1 - ((i + 1) / 2))) for i in x)
            for x in circuit_outputs
        ]

        return predicted_values


class QuantumKernelBinaryClassifier(QMLModel, ClassifierMixin):
    """
    This class implements the binary classifier based on quantum kernels.
    """

    def __init__(
        self,
        wires: Union[int, Sequence[int]],
        n_epochs: int = 10,
        kta_subset_size: int = 5,
        device: Optional[qml.Device] = None,
        optimizer: Optional[GradientDescentOptimizer] = None,
        embedding_method: Optional[Type[qml.operation.Operation]] = None,
        embedding_kwargs: Optional[Dict[str, Any]] = None,
        layers: Optional[Sequence[Type[qml.operation.Operation]]] = None,
        initial_weights: Optional[Sequence[float]] = None,
        rng_seed: int = 42,
        accuracy_threshold: float = 0.8,
        validation_set_size: float = 0.2,
        debug_flag: bool = True,
        coupling_map: Optional[Sequence[Sequence[int]]] = None,
    ) -> None:
        """
        A constructor for the `QuantumKernelBinaryClassifier` class.

        :param wires:
            The wires to use in the VQC or the number of qubits (and wires) used in the
            VQC.
        :param n_epochs:
            The maximal number of training iterations.
        :param kta_subset_size:
            The number of objects used to evaluate the kernel target alignment method
            in the cost function.
        :param device:
            A device on which the model should operate.
        :param optimizer:
            The optimizer that will be used in the training. `NesterovMomentumOptimizer`
            with default parameters will be used as default.
        :param embedding_method:
            Embedding method of the data. By default - when `None` is specified - the
            classifier will use `AngleEmbedding`. See `_prepare_default_embedding`
            for parameters details.
        :param embedding_kwargs:
            Keyword arguments for the embedding method. If none are specified the
            classifier will use the default embedding method.
        :param layers:
            A list of `layer` functions to be applied in the kernel ansatz VQC.
        :param initial_weights:
            The weights using which the training will start.
        :param rng_seed:
            A random seed used to initialize the weights (if no weights are given).
        :param accuracy_threshold:
            The accuracy after which the training is considered complete.
        :param validation_set_size:
            A part of the training set that will be used for classifier validation.
            It should be from (0, 1).
        :param debug_flag:
            A flag informing the classifier if the training info should be printed
            to the console or not.
        :param coupling_map:
            A description of connections between the qubits in the device.
        """
        super().__init__(
            wires,
            device,
            optimizer,
            embedding_method,
            embedding_kwargs,
            layers,
            validation_set_size,
            rng_seed,
            coupling_map,
        )

        self.n_epochs: int = n_epochs

        self._kta_subset_size: int = kta_subset_size

        self._accuracy_threshold = accuracy_threshold
        self._validation_set_size = validation_set_size

        self._weights_length: int = 0

        for layer in self._layers:
            self._weights_length += prod(
                layer.shape(n_layers=1, n_wires=len(self.wires))
            )

        self._rng_seed: int = rng_seed
        self._rng: random.Random = random.Random(rng_seed)

        if initial_weights is None or len(initial_weights) != self._weights_length:
            initial_weights = [
                np.pi * self._rng.random() for _ in range(self._weights_length)
            ]

        self.weights = np.array(initial_weights, requires_grad=True)

        self._debug_flag: bool = debug_flag

        self._classifier: SVC = SVC()

    def _ansatz(self, weights: Sequence[float], features: Sequence[float]) -> None:
        """
        A VQC ansatz that will be used in defining the quantum kernel function.

        :note:
            Templates expanding is necessary for the transpiling!

        :param weights:
            Weights that will be optimized during the learning process.
        :param features:
            Feature vector representing the object that is being classified.
        """

        start_weights: int = 0

        for layer in self._layers:
            for op in self._embedding_method(
                features, do_queue=False, **self._embedding_kwargs
            ).expand():
                qml.apply(op)

            layer_shape: Tuple[int, ...] = layer.shape(
                n_layers=1, n_wires=len(self.wires)
            )

            layer_weights = weights[start_weights : start_weights + prod(layer_shape)]
            start_weights += prod(layer_shape)
            layer_weights = np.array(layer_weights).reshape(layer_shape)

            for op in layer(layer_weights, do_queue=False, wires=self.wires).expand():
                qml.apply(op)

    def _create_transform(
        self,
    ) -> qml.QNode:
        """
        Creates a feature map VQC based on the current kernel.

        :return:
            The feature map VQC based on the current kernel.
        """

        # @qml.qnode(self.dev)
        def transform(
            weights: Sequence[float], features: Sequence[float]
        ) -> List[qml.measurements.ExpectationMP]:
            """
            The definition of the feature map VQC.

            :param weights:
                Parameters of the VQC.
            :param features:
                The features of the object to be transformed.

            :return:
                The result of `qml.PauliZ` measurements on the feature map VQC.
            """
            self._ansatz(weights, features)

            # TODO TR: Is this a good measurement to return?
            return [qml.expval(qml.PauliZ((i,))) for i in self.wires]

        if self.coupling_map:
            transform = transpile(coupling_map=self.coupling_map)(transform)

        return qml.QNode(transform, self.dev)

    def _create_kernel(
        self,
    ) -> Callable[[Sequence[float], Sequence[float], Sequence[float]], float]:
        """
        Prepares the VQC that will return the quantum kernel value for given data
        points.

        :return:
            The VQC structure representing the kernel function.
        """

        # Adjoint circuits is prepared pretty easily.
        adjoint_ansatz: Callable[
            [Sequence[float], Sequence[float]], None
        ] = qml.adjoint(self._ansatz)

        # @qml.qnode(self.dev)
        def kernel_circuit(
            weights: Sequence[float],
            first_features: Sequence[float],
            second_features: Sequence[float],
        ) -> qml.measurements.ProbabilityMP:
            """
            The VQC returning the quantum embedding kernel circuit based on given
            ansatz.

            :param weights:
                Weights to be applied to the VQC ansatz.
            :param first_features:
                Features of the first object.
            :param second_features:
                Features of the second objects.

            :return:
                The probability of observing respective computational-base states.
            """
            self._ansatz(weights, first_features)
            adjoint_ansatz(weights, second_features)
            return qml.probs(wires=self.wires)

        if self.coupling_map:
            kernel_circuit = transpile(coupling_map=self.coupling_map)(kernel_circuit)

        kernel_circuit = qml.QNode(kernel_circuit, device=self.dev)

        def kernel(
            weights: Sequence[float],
            first_features: Sequence[float],
            second_features: Sequence[float],
        ) -> float:
            """
            A method representing the quantum embedding kernel based on the given
            ansatz.

            :param weights:
                Weights to be applied to the VQC ansatz.
            :param first_features:
                Features of the first object.
            :param second_features:
                Features of the second objects.

            :return:
                The value of the kernel (or of measuring the zero state after running
                the VQC).
            """

            # The `np.array` casting is required so that indexing is "legal".
            return np.array(kernel_circuit(weights, first_features, second_features))[0]

        return kernel

    def _kernel_matrix_function(
        self, x: Sequence[Sequence[float]], y: Sequence[int]
    ) -> Callable[[Sequence[float], Sequence[float]], float]:
        """
        Prepares and returns the `kernel_matrix` function that uses the
        trained kernel.

        :param x:
            The lists of features of the objects that are used during the
            training.
        :param y:
            The classes corresponding to the given features.

        :return:
            The `kernel_matrix` function that uses the trained kernel.
        """
        kernel: Callable[
            [Sequence[float], Sequence[float], Sequence[float]], float
        ] = self._create_kernel()

        return qml.kernels.kernel_matrix(
            list(x),
            list(y),
            lambda x1, x2: kernel(self.weights, x1, x2),
        )

    def fit(
        self,
        X: Union[Sequence[Sequence[float]], NDArray[np.float64]],
        y: Optional[Sequence[ModelOutput]],
    ) -> "QuantumKernelBinaryClassifier":
        """
        The classifier training method.

        TODO TR: How to break this method down into smaller ones?

        :param X:
            The lists of features of the objects that are used during the training.
        :param y:
            A list of classes corresponding to the given lists of features. The classes
            should be from set {-1, 1}.

        :return:
            Returns `self` after training.
        """

        if not self.dev:
            raise AttributeError("Specify the device (dev) before fitting.")

        if y is None:
            raise AttributeError("Missing y in supervised learning model.")

        kernel: Callable[
            [Sequence[float], Sequence[float], Sequence[float]], float
        ] = self._create_kernel()

        self._split_data_for_training(X, y)

        def cost(weights: Sequence[float]) -> float:
            """
            A cost function used during the learning. We will use negative of kernel
            target alignment method, as the `pennylane` optimizers are meant for
            minimizing the objective function and `target_alignment` function is a
            similarity measure between the kernels.

            :param weights:
                Weights to be applied into kernel VQC.

            :return:
                The negative value of KTA for given weights.
            """

            # Choose subset of datapoints to compute the KTA on.
            subset: np.ndarray = np.array(
                self._rng.sample(
                    list(range(len(self._training_X))), self._kta_subset_size
                )
            )

            return -target_alignment(
                list(self._training_X[subset]),
                list(self._training_y[subset]),
                lambda x1, x2: self._create_kernel()(weights, x1, x2),
                assume_normalized_kernel=True,
            )

        for i in range(self.n_epochs):
            self.weights, cost_val = self.optimizer.step_and_cost(cost, self.weights)

            current_alignment = target_alignment(
                list(X),
                list(y),
                lambda x1, x2: kernel(self.weights, x1, x2),
                assume_normalized_kernel=True,
            )

            # Second create a kernel matrix function using the trained kernel.
            self._classifier = SVC(kernel=self._kernel_matrix_function).fit(X, y)

            accuracy: float = self.score(self._validation_X, self._validation_y)

            if self._debug_flag:
                print(
                    f"Step {i + 1}: "
                    f"Alignment = {current_alignment:.3f} | "
                    f"Step cost value = {cost_val} | "
                    f"Validation Accuracy {accuracy:.3f}"
                )
                print(self.weights)

            if accuracy >= self._accuracy_threshold:
                break

        return self

    def transform(
        self, features_lists: Sequence[Sequence[float]]
    ) -> List[List[qml.measurements.ExpectationMP]]:
        """
        Maps the object described by the `features` into it's representation in the
        feature space.

        :param features_lists:
            The features of the object to be mapped.

        :return:
            The representation of the given object in the feature space.
        """
        transform: qml.QNode = self._create_transform()

        mapped_features: List[List[qml.measurements.ExpectationMP]] = []

        for features_list in features_lists:
            mapped_features.append(transform(self.weights, features_list))

        return mapped_features

    def predict(
        self, features_lists: Sequence[Sequence[float]]
    ) -> Sequence[ModelOutput]:
        """
        Predicts and returns the classes of the objects for which features were given.
        It applies current `self.weights` as the parameters of VQC.

        :param features_lists:
            Objects' features to be encoded at the input of the VQC.

        :return:
            The results_reconstruction - classes 0 or 1 - of the classification.
            The data structure of the returned object is `np.ndarray` with `dtype=bool`.
        """
        return self._classifier.predict(features_lists)


class QNNClassifier(QMLModel, ClassifierMixin):
    """
    This class implements a quantum classifier based on the multiple binary quantum
    classifiers.
    """

    def __init__(
        self,
        wires: Union[int, Sequence[int]],
        n_classes: int,
        binary_classifiers: Optional[Sequence[QNNBinaryClassifier]] = None,
        batch_size: int = 10,
        accuracy_threshold: float = 0.8,
        device: Optional[qml.Device] = None,
        optimizer: Optional[GradientDescentOptimizer] = None,
        embedding_method: Optional[Type[qml.operation.Operation]] = None,
        embedding_kwargs: Optional[Dict[str, Any]] = None,
        layers: Optional[Sequence[Type[qml.operation.Operation]]] = None,
        validation_set_size: float = 0.2,
        rng_seed: int = 42,
    ) -> None:
        """
        The constructor for the `QuantumClassifier` class.

        :param wires:
            The wires to use in the VQC or the number of qubits (and wires) used in the
            VQC. It will be used in the `qml.Device` specification.
        :param n_classes:
            The number of classes in the classification task.
        :param binary_classifiers:
            Binary classifiers that will be used in the classification. If `None` is
            given, then the `QuantumClassifier` will produce default binary classifiers.
        :param batch_size:
            Batch size using during binary classifiers fitting.
        :param accuracy_threshold:
            The target minimal accuracy of the classifier. Note that it reflects total
            accuracy of the classifier, which is lower than accuracy of each binary
            classifier.
        :param device:
            A device on which the model should operate.
        :param optimizer:
            The optimizer that will be used in the training. `NesterovMomentumOptimizer`
            with default parameters will be used as default. It will be used in each of
            the binary classifiers initialized by `QuantumClassifier`.
        :param embedding_method:
            Embedding method of the data. By default - when `None` is specified - the
            QMLModel will use `AngleEmbedding`. See `_prepare_default_embedding`
            for parameters details. It will be used in each of the binary classifiers
            initialized by `QuantumClassifier`.
        :param embedding_kwargs:
            Keyword arguments for the embedding method. If `None` are specified the
            QMLModel will use the default embedding method. It will be used in each of
            the binary classifiers initialized by `QuantumClassifier`.
        :param layers:
            Layers to be used in the VQC. The layers will be applied in the given order.
            A double `StronglyEntanglingLayer` will be used if `None` is given.
        :param validation_set_size:
            A part of the training set that will be used for QMLModel validation.
            It should be from (0, 1). It will be used in each of the binary classifiers.
        :param rng_seed:
            A seed used for random weights initialization.
        """
        super().__init__(
            wires,
            device,
            optimizer,
            embedding_method,
            embedding_kwargs,
            layers,
            validation_set_size,
            rng_seed,
        )

        self.accuracy_threshold: float = accuracy_threshold
        self.n_classes = n_classes

        if not binary_classifiers:
            binary_classifiers = self._prepare_default_binary_classifiers(batch_size)

        self._binary_classifiers: Sequence[QNNBinaryClassifier] = binary_classifiers

    def _prepare_default_binary_classifiers(
        self, batch_size: int
    ) -> List[QNNBinaryClassifier]:
        """
        Prepares as set od default binary classifiers with the parameters specified
        during the class initialization.

        :param batch_size:
            Batch size to be used in the initialized binary classifiers.

        :return:
            List of the `QNNBinaryClassifier`s initialized with the given parameters.
        """
        binary_classifiers: List[QNNBinaryClassifier] = []

        for _ in range(self.n_classes):
            binary_classifiers.append(
                QNNBinaryClassifier(
                    wires=self.wires,
                    batch_size=batch_size,
                    embedding_method=self._embedding_method,
                    embedding_kwargs=self._embedding_kwargs,
                    layers=self._layers,
                    validation_set_size=self._validation_set_size,
                )
            )

        return binary_classifiers

    def seed(self, new_seed: int) -> None:
        """
        Sets up the new seed.

        :param new_seed:
            New seed to be applied to both the classifier and it's binary classifier
            parts.
        """
        self._rng_seed = new_seed

        for i in range(len(self._binary_classifiers)):
            self._binary_classifiers[i].seed(new_seed)

    def fit(
        self,
        X: Union[Sequence[Sequence[float]], NDArray[np.float64]],
        y: Optional[Sequence[ModelOutput]],
    ) -> "QNNClassifier":
        """
        The model training method. Essentially, it fits every binary classifier to the
        respective class.

        :param X:
            The lists of features of the objects that are used during the training.
        :param y:
            A list of outputs corresponding to the given lists of features.

        :return:
            Returns `self` after training.
        """

        if not self.dev:
            raise AttributeError("Specify the device (dev) before fitting.")

        if y is None:
            raise AttributeError("Missing y in supervised learning model.")

        # Check if there's a binary classifier for each class.
        unique_classes: np.ndarray = np.unique(y)

        if len(unique_classes) > len(self._binary_classifiers):
            raise AssertionError(
                "Numbers of provided binary classifiers and classes don't match!"
            )

        binary_classifier_accuracy_threshold: float = np.sqrt(self.accuracy_threshold)

        for classifier in self._binary_classifiers:
            classifier.accuracy_threshold = binary_classifier_accuracy_threshold

        # Fit each binary classifier to its respective class.
        for i in range(len(self._binary_classifiers)):
            classifier_classes = np.array(y)
            classifier_classes[classifier_classes != unique_classes[i]] = -1
            classifier_classes[classifier_classes == unique_classes[i]] = 1

            self._binary_classifiers[i].dev = self.dev
            self._binary_classifiers[i].fit(X, classifier_classes)

        return self

    def predict(self, features: Sequence[Sequence[float]]) -> Sequence[ModelOutput]:
        """
        Returns predictions of the model for the given features. In the case of
        `QuantumClassifier`, for given features, the predicted class corresponds to the
        index of the binary classifier which returns the highest expectation value of
        the `PauliZ` measurement on the first qubit.

        :param features:
            Features of the objects for which the model will predict the values.
        :return:
            Values predicted for given features.
        """
        predictions: List[int] = []

        for x in features:
            max_expectation: float = 0
            current_class: int = 0

            for i, classifier in enumerate(self._binary_classifiers):
                expectation: float = classifier.get_circuit_expectation_values([x])[0][
                    0
                ]

                if expectation > max_expectation:
                    max_expectation = expectation
                    current_class = i

            predictions.append(current_class)

        for classifier in self._binary_classifiers:
            classifier.circuit = None

        return predictions


class RBMClustering:
    """
    A class for performing clustering using Restricted Boltzmann Machine. The RBM can
    be trained using both classical and quantum algorithms. The quantum algorithm
    requires the D-Wave quantum annealer to be specified.
    """

    # TODO TR:  Remember to add the D-Wave handling.

    def __init__(
        self,
        lbae_input_shape: Tuple[int, ...],
        lbae_out_channels: int,
        lbae_n_layers: int,
        rmb_n_visible_neurons: int,
        rbm_n_hidden_neurons: int,
        n_gpus: int = 0,
        n_epochs: int = 100,
        learning_rate: float = 0.01,
        qubo_scale: float = 1.0,
        sampler: Optional[Sampler] = None,
        fireing_threshold: float = 0.8,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.lbae: LBAE = LBAE(
            input_size=lbae_input_shape[1:],  # TR: Notice shape reduction.
            out_channels=lbae_out_channels,
            latent_space_size=rmb_n_visible_neurons,
            num_layers=lbae_n_layers,
            quantize=True,  # Required, because it will be the input of the RBM.
        )

        self.rbm: RBM = RBM(
            num_visible=rmb_n_visible_neurons, num_hidden=rbm_n_hidden_neurons, rng=rng
        )

        self.n_gpus: int = n_gpus
        self.n_epochs: int = n_epochs
        self.learning_rate: float = learning_rate
        self.qubo_scale: float = qubo_scale

        self.sampler: Optional[Sampler] = sampler

        self.fireing_threshold: float = fireing_threshold

    def fit(
        self,
        data_loader: DataLoader[Tuple[Tensor, Tensor]],
    ) -> None:
        n_gpus: int = self.n_gpus if self.n_gpus > 0 else 1

        lbae_trainer: Trainer = Trainer(
            accelerator="cpu",  # TR TODO: Make it modifiable.
            num_nodes=n_gpus,
            max_epochs=self.n_epochs,
            deterministic=True,
        )

        print("LBAE training start.")
        lbae_trainer.fit(self.lbae, data_loader)
        print("LBAE training finished.")

        rbm_trainer: RBMTrainer
        # Pick a trainer depending on the existence of the sampler.
        if self.sampler is None:
            rbm_trainer = CD1Trainer(
                num_steps=self.n_epochs, learning_rate=self.learning_rate
            )
        else:
            rbm_trainer = AnnealingRBMTrainer(
                sampler=self.sampler,
                learning_rate=self.learning_rate,
                num_steps=self.n_epochs,
            )

        print("RBM training start.")
        rbm_trainer.fit(
            self.rbm, self._encoded_data_loader(data_loader, self.lbae.encoder)
        )
        print("RBM training finished.")

    @staticmethod
    def _encoded_data_loader(
        data_loader: DataLoader[Tuple[Tensor, Tensor]], encoder: LBAEEncoder
    ) -> Generator[Tuple[Tensor, Tensor], None, None]:
        """
        A generator that yields encoded data and targets from the given data loader. The
        data is encoded using the given encoder.

        :param data_loader:
            Data loader to be used for generating the encoded data.
        :param encoder:
            LBAE Encoder to be used for encoding the data.

        :return:
            Encoded data and targets.
        """
        while True:
            for _, (data, target) in enumerate(data_loader):
                yield encoder(data)[0], target

    def predict(self, x: Tensor) -> Tensor:
        """
        Returns the predicted class for the given input.

        :param x:
            Input to be classified.

        :return:
            Predicted class.
        """
        encoded_x: Tensor = self.lbae.encoder(x)[0]
        h_probs: NDArray[np.float32] = self.rbm.h_probs_given_v(
            encoded_x.detach().numpy()
        )[0]
        # print(h_probs)
        return Tensor([1 if p > self.fireing_threshold else 0 for p in h_probs])
