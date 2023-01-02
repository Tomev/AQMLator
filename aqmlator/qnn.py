"""
=============================================================================

    This module contains the functionalities related to the Quantum Neural Networks.

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

import torch
import pennylane as qml
from pennylane import numpy as np
from pennylane.templates.layers import StronglyEntanglingLayers
from pennylane.templates.embeddings import AmplitudeEmbedding
from pennylane.optimize import NesterovMomentumOptimizer, GradientDescentOptimizer
from typing import Sequence, Callable, Optional, Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from itertools import chain
from math import prod


class MLModel:
    """
    A boilerplate class, providing an interface for future ML models.
    """

    def __init__(
        self, optimizer: Optional[GradientDescentOptimizer] = None,
    ):
        """
        A constructor for MLModel `class`.
        :param optimizer:
            An optimizer that will be used during the training.
        """
        self._optimizer: GradientDescentOptimizer = optimizer
        self._weights: Sequence[float]

    @property
    def optimizer(self) -> GradientDescentOptimizer:
        """
        A getter for the optimizer.
        :return:
            The current optimizer.
        """
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer: GradientDescentOptimizer) -> None:
        """
        A setter for optimizer.
        :param optimizer:
            The new optimizer.
        """

    @property
    def weights(self) -> Sequence[float]:
        """
        A getter for the weights.
        :return:
            The current weights.
        """
        return self._weights

    @weights.setter
    def weights(self, weights: Sequence[float]) -> None:
        """
        A setter for the weights.

        :param weights:
            New weights.
        """
        self._weights = weights

    def fit(
        self, features_lists: Sequence[Sequence[float]], classes: Sequence[int]
    ) -> "MLModel":
        """
        The model training method.

        :param features_lists:
            The lists of features of the objects that are used during the training.
        :param classes:
            A list of classes corresponding to the given lists of features.

        :return:
            Returns self after training.
        """
        raise NotImplementedError


class QMLModel(MLModel):
    """
    A boilerplate class, providing an interface for future QML models.
    """

    def n_executions(self) -> int:
        """
        Returns number of VQC executions so far.

        :return:
            Returns the number of times the quantum device was called.
        """
        raise NotImplementedError


class QNNBinaryClassifier(QMLModel):
    """
    This class implements a binary classifier that uses Quantum Neural Networks.

    The classifier expects two classes given by ints 0 and 1.
    """

    def __init__(
        self,
        n_qubits: int,
        batch_size: int,
        n_epochs: int = 1,
        device_string: str = "lightning.qubit",
        optimizer: Optional[GradientDescentOptimizer] = None,
        embedding_method: Optional[qml.operation.Operation] = None,
        embedding_kwargs: Optional[Dict[str, Any]] = None,
        layers: Optional[Sequence[qml.operation.Operation]] = None,
        layers_weights_shapes: Optional[Sequence[Tuple[int, ...]]] = None,
        accuracy_threshold: float = 0.8,
        initial_weights: Optional[Sequence[float]] = None,
        weights_random_seed: int = 42,
        debug_flag: bool = True,
    ) -> None:
        """
        The constructor for the `QNNBinaryClassifier` class.

        :param n_qubits:
            The number of qubits (and wires) used in the classification.
        :param n_layers:
            The number of layers in the VQC.
        :param batch_size:
            Size of a batches used during the training.
        :param n_epochs:
            The number of training epochs.
        :param device_string:
            A string naming the device used to run the VQC.
        :param optimizer:
            The optimizer that will be used in the training. `NesterovMomentumOptimizer`
            with default parameters will be used as default.
        :param embedding_method:
            Embedding method of the data. By default - when `None` is specified - the
            classifier will use `AmplitudeEmbedding`. See `_prepare_default_embedding`
            for parameters details.
        :param embedding_kwargs:
            Keyword arguments for the embedding method. If none are specified the
            classifier will use the default embedding method.
        :param layers:
            Layers to be used in the VQC. The layers will be applied in the given order.
            A single `StronglyEntanglingLayer` will be used if `Ńone` is given.
        :param layers_weights_shapes:
            The shapes of the corresponding layers. Note that the default layers setup
            will be used if `None` is given.
        :param accuracy_threshold:
            The satisfactory accuracy of the classifier.
        :param initial_weights:
            The initial weights for the training.
        :param weights_random_seed:
            A seed used for random weights initialization.
        :param debug_flag:
            A flag informing the classifier if the training info should be printed
            to the console or not.
        """
        self._n_qubits: int = n_qubits
        self._n_epochs: int = n_epochs
        self._batch_size: int = batch_size
        self._dev_str: str = device_string

        self._accuracy_threshold: float = accuracy_threshold

        self._layers: Sequence[qml.operation.Operation]
        self._layers_weights_shapes: Sequence[Tuple[int, ...]]

        if layers is None or layers_weights_shapes is None:
            self._prepare_default_layers()
        elif len(layers) != len(layers_weights_shapes):
            self._prepare_default_layers()
        else:
            self._layers = layers
            self._layers_weights_shapes = layers_weights_shapes

        self._weights_length: int = 0

        for shape in self._layers_weights_shapes:
            self._weights_length += prod(shape)

        if initial_weights is None or len(initial_weights) != self._weights_length:
            np.random.seed(weights_random_seed)
            initial_weights = [
                np.pi * np.random.rand() for _ in range(self._weights_length)
            ]

        self._weights = initial_weights

        self._embedding_method: qml.operation.Operation
        self._embedding_kwargs: Dict[str, Any]

        if embedding_method is None or embedding_kwargs is None:
            self._prepare_default_embedding()
        else:
            self._embedding_method = embedding_method
            self._embedding_kwargs = embedding_kwargs

        if optimizer is None:
            optimizer = NesterovMomentumOptimizer()

        super().__init__(optimizer)

        # self._optimizer: GradientDescentOptimizer = optimizer

        self._dev = qml.device(device_string, wires=self._n_qubits)

        self._circuit: Callable[
            [Sequence[float], Sequence[float]], float
        ] = self._create_circuit()

        self._debug_flag: bool = debug_flag

    @property
    def n_qubits(self) -> int:
        """
        A getter for n_qubits property.

        :return:
            The current number of qubits in the device.
        """
        return self._n_qubits

    @n_qubits.setter
    def n_qubits(self, n_qubits: int) -> None:
        """
        A setter for n_qubits property.

        :Note: It also resets the device.

        :param n_qubits:
            The new number of qubits in the device.
        """
        self._n_qubits = n_qubits
        self._dev = qml.device(self._dev_str, wires=self._n_qubits)

    @property
    def n_epochs(self) -> int:
        """
        A getter for `n_qubits` property.

        :return:
            Returns the currently set number of epochs.
        """
        return self._n_epochs

    @n_epochs.setter
    def n_epochs(self, n_epochs: int) -> None:
        """
        A setter for `n_epochs` property.

        :param n_epochs:
            The new number of epochs.
        """
        self._n_epochs = n_epochs

    def n_executions(self) -> int:
        """
        Returns number of VQC executions so far.

        :return:
            Returns the number of times the quantum device was called.
        """
        return self._dev.num_executions

    @property
    def weights(self) -> Sequence[float]:
        """
        A getter for the `weights` property.

        :return:
            The current weights.
        """
        return self._weights

    @weights.setter
    def weights(self, weights: Sequence[float]) -> None:
        """
        A setter for the `weights` property.

        :param weigths:
            A new set of weights for the VQC.
        """
        self._weights = np.array(weights, requires_grad=True)

    def _prepare_default_embedding(self) -> None:
        """
        Prepares the default embedding method is `None` was specified or if the
        kwargs were `None`. The default one is simple `AmplitudeEmbedding`.
        """
        self._embedding_method = AmplitudeEmbedding
        self._embedding_kwargs = {
            "wires": range(self._n_qubits),
            "pad_with": 0,
            "normalize": True,
        }

    def _prepare_default_layers(self) -> None:
        """
        Prepares the default layers of the classifier if `None` was given in either
        `layers` or `layers_weights_number` arguments of the constructor. We will
        use a single strongly entangling layer.
        """
        self._layers = [StronglyEntanglingLayers]
        self._layers_weights_shapes = [(1, self._n_qubits, 3)]

    def _create_circuit(self) -> Callable[[Sequence[float], Sequence[float]], float]:
        @qml.qnode(self._dev)
        def circuit(weights: Sequence[float], features: Sequence[float]) -> float:
            """
            Returns the expectation value of the first qubit of the VQC of which the
            weights are optimized during the learning process.

            :param weights:
                Weights that will be optimized during the learning process.
            :param features:
                Feature vector representing the object that is being classified.

            :return:
                The expectation value (from range[-1, 1]) of the measurement in the
                computational basis of given circuit. This value is interpreted as
                the classification result and its confidence.
            """

            # TODO TR:  This is exactly how it's called, eg.
            #           `AmplitudeEmbedding(features, **kwargs)`.
            #           I need to figure out how to handle this warning.
            self._embedding_method(features, **self._embedding_kwargs)

            start_weights: int = 0

            for i, layer in enumerate(self._layers):
                layer_weights = weights[
                    start_weights : start_weights + prod(self._layers_weights_shapes[i])
                ]
                start_weights += prod(self._layers_weights_shapes[i])
                layer_weights = np.array(layer_weights).reshape(
                    self._layers_weights_shapes[i]
                )
                layer(layer_weights, wires=range(self._n_qubits))
            return qml.expval(qml.PauliZ((0,)))

        return circuit

    def cost(
        self,
        weights: Sequence[float],
        features_lists: Sequence[Sequence[float]],
        classes: Sequence[int],
    ) -> float:
        """
        Evaluates and returns the cost function value for the training purposes.

        :param features_lists:
            The lists of features of the objects that are being classified during the
            training.

        :param classes:
            Classes corresponding to the given features.

        :return:
            The value of the square loss function.
        """
        expectation_values: np.ndarray = np.array(
            [self._circuit(weights, x) for x in features_lists]
        )
        return np.mean((expectation_values - np.array(classes)) ** 2)

    def fit(
        self, features_lists: Sequence[Sequence[float]], classes: Sequence[int]
    ) -> "QNNBinaryClassifier":
        """
        The classifier training method.

        TODO TR: How to break this method down into smaller ones?

        :param features_lists:
            The lists of features of the objects that are used during the training.
        :param classes:
            A list of classes corresponding to the given lists of features.

        :return:
            Returns `self` after training.
        """
        train_features: Sequence[Sequence[float]]
        validation_features: Sequence[Sequence[float]]

        train_classes: Sequence[int]
        validation_classes: Sequence[int]

        (
            train_features,
            validation_features,
            train_classes,
            validation_classes,
        ) = train_test_split(features_lists, classes, test_size=0.2)

        # Change classes to [-1, 1]. See the method description for the reasoning.
        cost_classes: np.ndarray = np.array(train_classes)
        cost_classes[cost_classes == max(cost_classes)] = 1
        cost_classes[cost_classes == min(cost_classes)] = -1

        n_batches: int = max(1, len(features_lists) // self._batch_size)

        feature_batches = np.array_split(np.arange(len(train_features)), n_batches)

        best_weights: Sequence[float] = self._weights
        best_accuracy: float = 0

        self._weights = np.array(self._weights, requires_grad=True)
        cost: float
        batch_indices: np.tensor  # Of ints.

        for it, batch_indices in enumerate(
            chain(*(self._n_epochs * [feature_batches]))
        ):
            # Update the weights by one optimizer step
            def batch_cost(weights: Sequence[float]):
                return self.cost(
                    weights, train_features[batch_indices], cost_classes[batch_indices],
                )

            self._weights, cost = self._optimizer.step_and_cost(
                batch_cost, self._weights
            )

            # Compute accuracy on train and validation set
            accuracy_train = self.score(
                train_features[batch_indices], train_classes[batch_indices]
            )
            accuracy_validation = self.score(validation_features, validation_classes)

            # Make decision about stopping the training basing on the validation score
            if accuracy_validation >= best_accuracy:
                best_accuracy = accuracy_validation
                best_weights = self._weights

            if self._debug_flag:
                print(
                    f"It: {it + 1} / {self._n_epochs * n_batches} | Cost: {cost} |"
                    f" Accuracy (train): {accuracy_train} |"
                    f" Accuracy (validation): {accuracy_validation}"
                )

            if accuracy_train >= self._accuracy_threshold:
                break

        self._weights = best_weights

        return self

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
        expectation_values: np.ndarray = np.zeros(len(features_lists), dtype=float)

        for i, features in enumerate(features_lists):
            expectation_values[i] = self._circuit(self._weights, features)

        return expectation_values

    def predict(self, features_lists: Sequence[Sequence[float]]) -> np.ndarray:
        """
        Predicts and returns the classes of the objects for which features were given.
        It applies current `self.weights` as the parameters of VQC.

        :param features_lists:
            Objects' features to be encoded at the input of the VQC.

        :return:
            The results - classes 0 or 1 - of the classification. The data structure of
            the returned object is `np.ndarray` with `dtype=bool`.
        """
        return self.get_circuit_expectation_values(features_lists) >= 0.0

    def score(
        self,
        features_lists: Sequence[Sequence[float]],
        classes: Sequence[int],
        weights: Optional[Sequence[float]] = None,
    ) -> float:
        """
        Computes and returns the mean score of the classifier.

        :param features_lists:
            Features of the objects of interest.
        :param classes:
            The true target classes of given objects.
        :param weights:
            Weights that will be applied to the quantum circuit.

        :return:
            The mean score of the classifier.
        """
        if weights is not None:
            self._weights = weights

        score: float = 0

        predictions: np.ndarray = self.predict(features_lists)

        for i in range(len(features_lists)):
            score += int(predictions[i] == classes[i])  # Implicit conversion to int.

        return score / len(features_lists)

    def get_torch_layer(self) -> torch.nn.Module:
        """
        This method creates a PyTorch (quantum) layer based on the VQC.

        TODO TR:
            This method uses the same `circuit` method as the `_create_circuit`
            method. Perhaps this could be shared in some way.

            :return:
                Returns a PyTorch Layer made from the VQC.
        """

        @qml.qnode(self._dev, interface="torch")
        def circuit(inputs: torch.Tensor, weights: torch.Tensor) -> float:
            """
            Returns the expectation value of the first qubit of the VQC of which the
            weights are optimized during the learning process.

            :param inputs:
                Feature vector representing the object that is being classified. It
                has to be named `inputs`, otherwise there are errors occurring.

            :param weights:
                Weights that will be optimized during the learning process.

            :return:
                The expectation value (from range[-1, 1]) of the measurement in the
                computational basis of given circuit. This value is interpreted as
                the classification result and its confidence.
            """

            padding: torch.Tensor = torch.zeros([2 ** self._n_qubits - len(inputs)])
            inputs = torch.cat((inputs, padding))

            # TODO TR:  This is exactly how it's called, eg.
            #           `AmplitudeEmbedding(features, **kwargs)`.
            #           I need to figure out how to handle this warning.
            self._embedding_method(inputs, **self._embedding_kwargs)

            start_weights: int = 0

            for i, layer in enumerate(self._layers):
                layer_weights = weights[
                    start_weights : start_weights + prod(self._layers_weights_shapes[i])
                ]
                start_weights += prod(self._layers_weights_shapes[i])
                layer_weights = layer_weights.reshape(self._layers_weights_shapes[i])
                layer(layer_weights, wires=range(self._n_qubits))

            return qml.expval(qml.PauliZ((0,)))

        weight_shapes: Dict[str, int] = {"weights": len(self._weights)}
        return qml.qnn.TorchLayer(circuit, weight_shapes)
