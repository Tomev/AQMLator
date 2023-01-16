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

import torch
import pennylane as qml
from pennylane import numpy as np
from pennylane.templates.layers import StronglyEntanglingLayers
from pennylane.templates.embeddings import AmplitudeEmbedding, AngleEmbedding
from pennylane.optimize import NesterovMomentumOptimizer, GradientDescentOptimizer
from pennylane.kernels import target_alignment
from typing import Sequence, Callable, Optional, Dict, Any, Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.base import ClassifierMixin
from itertools import chain
from math import prod
import abc
import random


class MLModel(abc.ABC):
    """
    A boilerplate class, providing an interface for future ML models.
    """

    def __init__(
        self,
        optimizer: Optional[GradientDescentOptimizer] = None,
    ):
        """
        A constructor for MLModel `class`.
        :param optimizer:
            An optimizer that will be used during the training.
        """
        self.optimizer: GradientDescentOptimizer = optimizer
        self.weights: Sequence[float]

    @abc.abstractmethod
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


class QMLModel(MLModel, abc.ABC):
    """
    A boilerplate class, providing an interface for future QML models.
    """

    def __init__(self, optimizer: Optional[GradientDescentOptimizer] = None):
        self._dev: qml.Device
        super().__init__(optimizer)

    def n_executions(self) -> int:
        """
        Returns number of VQC executions so far.

        :return:
            Returns the number of times the quantum device was called.
        """
        return self._dev.num_executions


class QNNBinaryClassifier(QMLModel, ClassifierMixin):
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
        rng_seed: int = 42,
        validation_set_size: float = 0.2,
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
            A single `StronglyEntanglingLayer` will be used if `None` is given.
        :param layers_weights_shapes:
            The shapes of the corresponding layers. Note that the default layers setup
            will be used if `None` is given.
        :param accuracy_threshold:
            The satisfactory accuracy of the classifier.
        :param initial_weights:
            The initial weights for the training.
        :param rng_seed:
            A seed used for random weights initialization.
        :param validation_set_size:
            A part of the training set that will be used for classifier validation.
            It should be from (0, 1).
        :param debug_flag:
            A flag informing the classifier if the training info should be printed
            to the console or not.
        """
        self.n_qubits: int = n_qubits
        self._n_epochs: int = n_epochs
        self._batch_size: int = batch_size
        self._dev_str: str = device_string
        self._rng_seed: int = rng_seed

        self._accuracy_threshold: float = accuracy_threshold
        self._validation_set_size: float = validation_set_size

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

        self._rng: random.Random = random.Random(rng_seed)

        if initial_weights is None or len(initial_weights) != self._weights_length:
            initial_weights = [
                np.pi * self._rng.random() for _ in range(self._weights_length)
            ]

        self.weights = initial_weights

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

        self._dev = qml.device(device_string, wires=self.n_qubits)

        self._circuit: Callable[
            [Sequence[float], Sequence[float]], float
        ] = self._create_circuit()

        self._debug_flag: bool = debug_flag

    def _prepare_default_embedding(self) -> None:
        """
        Prepares the default embedding method is `None` was specified or if the
        kwargs were `None`. The default one is simple `AmplitudeEmbedding`.
        """
        self._embedding_method = AmplitudeEmbedding
        self._embedding_kwargs = {
            "wires": range(self.n_qubits),
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
        self._layers_weights_shapes = [(1, self.n_qubits, 3)]

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
                The expectation value (from range [-1, 1]) of the measurement in the
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
                layer(layer_weights, wires=range(self.n_qubits))
            return qml.expval(qml.PauliZ((0,)))

        return circuit

    def _cost(
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
        ) = train_test_split(
            features_lists,
            classes,
            test_size=self._validation_set_size,
            random_state=self._rng_seed,
        )

        # Change classes to [-1, 1]. See the method description for the reasoning.
        cost_classes: np.ndarray = np.array(train_classes)
        cost_classes[cost_classes == max(cost_classes)] = 1
        cost_classes[cost_classes == min(cost_classes)] = -1

        n_batches: int = max(1, len(features_lists) // self._batch_size)

        feature_batches = np.array_split(np.arange(len(train_features)), n_batches)

        best_weights: Sequence[float] = self.weights
        best_accuracy: float = 0

        self.weights = np.array(self.weights, requires_grad=True)
        cost: float
        batch_indices: np.tensor  # Of ints.

        def batch_cost(weights: Sequence[float]):
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
                train_features[batch_indices],
                cost_classes[batch_indices],
            )

        for it, batch_indices in enumerate(
            chain(*(self._n_epochs * [feature_batches]))
        ):
            # Update the weights by one optimizer step

            self.weights, cost = self.optimizer.step_and_cost(batch_cost, self.weights)

            # Compute accuracy on the validation set
            accuracy_validation = self.score(validation_features, validation_classes)

            # Make decision about stopping the training basing on the validation score
            if accuracy_validation >= best_accuracy:
                best_accuracy = accuracy_validation
                best_weights = self.weights

            if self._debug_flag:
                print(
                    f"It: {it + 1} / {self._n_epochs * n_batches} | Cost: {cost} |"
                    f" Accuracy (validation): {accuracy_validation}"
                )

            if accuracy_validation >= self._accuracy_threshold:
                break

        self.weights = best_weights

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
            expectation_values[i] = self._circuit(self.weights, features)

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

            padding: torch.Tensor = torch.zeros([2**self.n_qubits - len(inputs)])
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
                layer(layer_weights, wires=range(self.n_qubits))

            return qml.expval(qml.PauliZ((0,)))

        weight_shapes: Dict[str, int] = {"weights": len(self.weights)}
        return qml.qnn.TorchLayer(circuit, weight_shapes)


class QuantumKernelBinaryClassifier(QMLModel, ClassifierMixin):
    """
    This class implements the binary classifier based on quantum kernels.
    """

    def __init__(
        self,
        n_qubits: int,
        n_epochs: int = 10,
        kta_subset_size: int = 5,
        device_string: str = "lightning.qubit",
        optimizer: Optional[GradientDescentOptimizer] = None,
        embedding_method: Optional[qml.operation.Operation] = None,
        embedding_kwargs: Optional[Dict[str, Any]] = None,
        layers: Optional[Sequence[qml.operation.Operation]] = None,
        layers_weights_shapes: Optional[Sequence[Tuple[int, ...]]] = None,
        initial_weights: Optional[Sequence[float]] = None,
        rng_seed: int = 42,
        accuracy_threshold: float = 0.8,
        validation_set_size: float = 0.2,
        debug_flag: bool = True,
    ) -> None:
        """
        A constructor for the `QuantumKernelBinaryClassifier` class.

        :param n_qubits:
            The number of qubits in the kernel VQC.
        :param n_epochs:
            The maximal number of training iterations.
        :param kta_subset_size:
            The number of objects used to evaluate the kernel target alignment method
            in the cost function.
        :param device_string:
            A string naming the device used to run the VQC.
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
        :param layers_weights_shapes:
            The parameters for the `layer` methods corresponding to the `layers`.
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
        """
        self._dev: qml.Device = qml.device(device_string, wires=n_qubits)

        self.n_qubits: int = n_qubits
        self.n_epochs: int = n_epochs

        self._kta_subset_size: int = kta_subset_size

        self._accuracy_threshold = accuracy_threshold
        self._validation_set_size = validation_set_size

        self._layers: Sequence[qml.operation.Operation]
        self._layers_weights_shapes: Sequence[Tuple[int, ...]]

        if layers is None or layers_weights_shapes is None:
            self._prepare_default_layers()
        elif len(layers) != len(layers_weights_shapes):
            self._prepare_default_layers()
        else:
            self._layers = layers
            self._layers_weights_shapes = layers_weights_shapes

        self._embedding_method: qml.operation.Operation
        self._embedding_kwargs: Dict[str, Any]

        if embedding_method is None or embedding_kwargs is None:
            self._prepare_default_embedding()
        else:
            self._embedding_method = embedding_method
            self._embedding_kwargs = embedding_kwargs

        self._weights_length: int = 0

        for shape in self._layers_weights_shapes:
            self._weights_length += prod(shape)

        self._rng_seed: int = rng_seed
        self._rng: random.Random = random.Random(rng_seed)

        if initial_weights is None or len(initial_weights) != self._weights_length:
            initial_weights = [
                np.pi * self._rng.random() for _ in range(self._weights_length)
            ]

        self.weights = np.array(initial_weights, requires_grad=True)

        if optimizer is None:
            optimizer = NesterovMomentumOptimizer()

        self._debug_flag: bool = debug_flag

        super().__init__(optimizer)

        self._classifier: SVC = SVC()

    def _prepare_default_embedding(self) -> None:
        """
        Prepares the default embedding method is `None` was specified or if the
        kwargs were `None`. The default one is simple `AngleEmbedding`.

        Note:
            We use `AngleEmbedding` here, because we will apply the embedding multiple
            times (at the beginning of each layer), which cannot be done with
            `AmplitudeEmbedding`.
        """
        self._embedding_method = AngleEmbedding
        self._embedding_kwargs = {"wires": range(self.n_qubits)}

    def _prepare_default_layers(self) -> None:
        """
        Prepares the default layers of the classifier if `None` was given in either
        `layers` or `layers_weights_number` arguments of the constructor. We will
        use a triple strongly entangling layer.
        """
        self._layers = [StronglyEntanglingLayers] * 3
        self._layers_weights_shapes = [(1, self.n_qubits, 3)] * 3

    def _ansatz(self, weights: Sequence[float], features: Sequence[float]) -> None:
        """
        A VQC ansatz that will be used in defining the quantum kernel function.

        :param weights:
            Weights that will be optimized during the learning process.
        :param features:
            Feature vector representing the object that is being classified.
        """

        start_weights: int = 0

        for i, layer in enumerate(self._layers):
            # TODO TR:  This is exactly how it's called, eg.
            #           `AmplitudeEmbedding(features, **kwargs)`.
            #           I need to figure out how to handle this warning.
            self._embedding_method(features, **self._embedding_kwargs)

            layer_weights = weights[
                start_weights : start_weights + prod(self._layers_weights_shapes[i])
            ]
            start_weights += prod(self._layers_weights_shapes[i])
            layer_weights = np.array(layer_weights).reshape(
                self._layers_weights_shapes[i]
            )
            layer(layer_weights, wires=range(self.n_qubits))

    def _create_transform(
        self,
    ) -> Callable[
        [Sequence[float], Sequence[float]], List[qml.measurements.ExpectationMP]
    ]:
        """
        Creates a feature map VQC based on the current kernel.

        :return:
            The feature map VQC based on the current kernel.
        """

        @qml.qnode(self._dev)
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
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        return transform

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

        @qml.qnode(self._dev)
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
            return qml.probs(wires=range(self.n_qubits))

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

    def fit(
        self, features_lists: Sequence[Sequence[float]], classes: Sequence[int]
    ) -> "QuantumKernelBinaryClassifier":
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

        kernel: Callable[
            [Sequence[float], Sequence[float], Sequence[float]], float
        ] = self._create_kernel()

        train_features: Sequence[Sequence[float]]
        validation_features: Sequence[Sequence[float]]

        train_classes: Sequence[int]
        validation_classes: Sequence[int]

        (
            train_features,
            validation_features,
            train_classes,
            validation_classes,
        ) = train_test_split(
            features_lists,
            classes,
            test_size=self._validation_set_size,
            random_state=self._rng_seed,
        )

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
                    list(range(len(train_features))), self._kta_subset_size
                )
            )

            return -target_alignment(
                list(train_features[subset]),
                list(train_classes[subset]),
                lambda x1, x2: self._create_kernel()(weights, x1, x2),
                assume_normalized_kernel=True,
            )

        for i in range(self.n_epochs):

            self.weights, cost_val = self.optimizer.step_and_cost(cost, self.weights)

            current_alignment = target_alignment(
                list(features_lists),
                list(classes),
                lambda x1, x2: kernel(self.weights, x1, x2),
                assume_normalized_kernel=True,
            )

            # Second create a kernel matrix function using the trained kernel.
            def kernel_matrix_function(
                features_lists: Sequence[Sequence[float]], classes: Sequence[int]
            ) -> Callable[[Sequence[float], Sequence[float]], float]:
                """
                Prepares and returns the `kernel_matrix` function that uses the
                trained kernel.

                :param features_lists:
                    The lists of features of the objects that are used during the
                    training.
                :param classes:
                    The classes corresponding to the given features.

                :return:
                    The `kernel_matrix` function that uses the trained kernel.
                """
                return qml.kernels.kernel_matrix(
                    features_lists,
                    classes,
                    lambda x1, x2: kernel(self.weights, x1, x2),
                )

            self._classifier = SVC(kernel=kernel_matrix_function).fit(
                features_lists, classes
            )

            accuracy: float = self.score(validation_features, validation_classes)

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
        transform: Callable[
            [Sequence[float], Sequence[float]], List[qml.measurements.ExpectationMP]
        ] = self._create_transform()

        mapped_features: List[List[qml.measurements.ExpectationMP]] = []

        for features_list in features_lists:
            mapped_features.append(transform(self.weights, features_list))

        return mapped_features

    def predict(self, features_lists: Sequence[Sequence[float]]) -> Sequence[int]:
        """
        Predicts and returns the classes of the objects for which features were given.
        It applies current `self.weights` as the parameters of VQC.

        :param features_lists:
            Objects' features to be encoded at the input of the VQC.

        :return:
            The results - classes 0 or 1 - of the classification. The data structure of
            the returned object is `np.ndarray` with `dtype=bool`.
        """
        return self._classifier.predict(features_lists)
