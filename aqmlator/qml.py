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
import torch
import random

import pennylane as qml

from math import prod
from itertools import chain
from typing import Sequence, Callable, Optional, Dict, Any, Tuple, List, Type, Union

from sklearn.svm import SVC
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.model_selection import train_test_split

from pennylane import numpy as np
from pennylane.templates.layers import StronglyEntanglingLayers
from pennylane.templates.embeddings import AmplitudeEmbedding, AngleEmbedding
from pennylane.optimize import NesterovMomentumOptimizer, GradientDescentOptimizer
from pennylane.kernels import target_alignment


class QMLModel(abc.ABC):
    """
    A boilerplate class, providing an interface for future QML models.
    """

    def __init__(
        self,
        wires: Union[int, Sequence[int]],
        device_string: str = "lightning.qubit",
        optimizer: Optional[GradientDescentOptimizer] = None,
        embedding_method: Optional[Type[qml.operation.Operation]] = None,
        embedding_kwargs: Optional[Dict[str, Any]] = None,
        layers: Optional[Sequence[Type[qml.operation.Operation]]] = None,
        layers_weights_shapes: Optional[Sequence[Tuple[int, ...]]] = None,
        validation_set_size: float = 0.2,
        rng_seed: int = 42,
    ):
        """
        The constructor for the `QMLModel` class.

        :param wires:
            The wires to use in the VQC or the number of qubits (and wires) used in the
            VQC.
        :param device_string:
            A string naming the device used to run the VQC.
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
        :param layers_weights_shapes:
            The shapes of the corresponding layers. Note that the default layers setup
            will be used if `None` is given.
        :param validation_set_size:
            A part of the training set that will be used for QMLModel validation.
            It should be from (0, 1).
        :param rng_seed:
            A seed used for random weights initialization.
        """
        self._dev: qml.Device
        self._device_string: str = device_string

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
        self._layers_weights_shapes: Sequence[Tuple[int, ...]]

        if layers is None or layers_weights_shapes is None:
            self._prepare_default_layers()
        elif len(layers) != len(layers_weights_shapes):
            self._prepare_default_layers()
        else:
            self._layers = layers
            self._layers_weights_shapes = layers_weights_shapes

        self._rng_seed: int = rng_seed
        self._rng: random.Random = random.Random(rng_seed)

        self._validation_set_size: float = validation_set_size

        if optimizer is None:
            optimizer = NesterovMomentumOptimizer()

        self.optimizer: GradientDescentOptimizer = optimizer
        self.weights: Sequence[float]

    def n_executions(self) -> int:
        """
        Returns number of VQC executions so far.

        :return:
            Returns the number of times the quantum device was called.
        """
        return self._dev.num_executions

    @abc.abstractmethod
    def fit(
        self,
        features_lists: Sequence[Sequence[float]],
        classes: Sequence[int],
        dev: qml.Device,
    ) -> "QMLModel":
        """
        The model training method.

        :param features_lists:
            The lists of features of the objects that are used during the training.
        :param classes:
            A list of classes corresponding to the given lists of features.
        :param dev:
            A quantum device on which the fitting should be performed. If `None` then
            a new qml.Device will be initialised.

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
        self._layers_weights_shapes = [(1, len(self.wires), 3)] * 2

    def _split_data_for_training(
        self, X: Sequence[Sequence[float]], y: Sequence[int]
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
        device_string: str = "lightning.qubit",
        optimizer: Optional[GradientDescentOptimizer] = None,
        embedding_method: Optional[Type[qml.operation.Operation]] = None,
        embedding_kwargs: Optional[Dict[str, Any]] = None,
        layers: Optional[Sequence[Type[qml.operation.Operation]]] = None,
        layers_weights_shapes: Optional[Sequence[Tuple[int, ...]]] = None,
        accuracy_threshold: float = 0.8,
        initial_weights: Optional[Sequence[float]] = None,
        rng_seed: int = 42,
        validation_set_size: float = 0.2,
        prediction_function: Optional[
            Callable[[Sequence[float]], Union[Sequence[float], Sequence[int]]]
        ] = None,
        debug_flag: bool = True,
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
        :param device_string:
            A string naming the device used to run the VQC.
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
        :param layers_weights_shapes:
            The shapes of the corresponding layers. Note that the default layers setup
            will be used if `None` is given.
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
        """
        super().__init__(
            wires,
            device_string,
            optimizer,
            embedding_method,
            embedding_kwargs,
            layers,
            layers_weights_shapes,
            validation_set_size,
            rng_seed,
        )

        self._n_epochs: int = n_epochs
        self._batch_size: int = batch_size
        self._dev_str: str = device_string

        self._accuracy_threshold: float = accuracy_threshold

        self._weights_length: int = 0

        for shape in self._layers_weights_shapes:
            self._weights_length += prod(shape)

        if initial_weights is None or len(initial_weights) != self._weights_length:
            initial_weights = [
                np.pi * self._rng.random() for _ in range(self._weights_length)
            ]

        self.weights = initial_weights

        self._dev = qml.device(device_string, wires=self.wires)

        self._circuit: Callable[
            [Sequence[float], Sequence[float]], Sequence[float]
        ] = self._create_circuit()

        self._debug_flag: bool = debug_flag

        if prediction_function is None:
            prediction_function = self._default_prediction_function

        self._prediction_function: Callable[
            [Sequence[float]], Sequence[float]
        ] = prediction_function

    def _create_circuit(
        self, interface: str = "autograd"
    ) -> Callable[[Sequence[float], Sequence[float]], Sequence[float]]:
        @qml.qnode(self._dev, interface=interface)
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

            self._embedding_method(inputs, **self._embedding_kwargs)

            start_weights: int = 0

            for i, layer in enumerate(self._layers):
                layer_weights = weights[
                    start_weights : start_weights + prod(self._layers_weights_shapes[i])
                ]

                start_weights += prod(self._layers_weights_shapes[i])

                layer_weights = layer_weights.reshape(self._layers_weights_shapes[i])

                layer(layer_weights, wires=self.wires)

            return [qml.expval(qml.PauliZ((i))) for i in self.wires]
            # return qml.expval(qml.PauliZ((self.wires[0],)))

        return circuit

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

        expectation_values: List[Sequence[float]] = []

        for i, features in enumerate(features_lists):
            expectation_values.append(self._circuit(features, np.array(self.weights)))

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
        raise NotImplemented

    @abc.abstractmethod
    def _default_prediction_function(
        self, circuit_outputs: Sequence[Sequence[float]]
    ) -> Union[Sequence[float], Sequence[int]]:
        """
        The default prediction function that should be specified for every QNN-based
        model.

        :param circuit_output:
            The output of the VQC.

        :return:
            Returns prediction value for the given problem.
        """

    def fit(
        self, X: Sequence[Sequence[float]], y: Sequence[int], dev: qml.Device = None
    ) -> "QNNModel":
        """
        The model training method.

        TODO TR: How to break this method down into smaller ones?

        :param X:
            The lists of features of the objects that are used during the training.
        :param y:
            A list of outputs corresponding to the given lists of features.
        :param dev:
            A quantum device on which the fitting should be performed. If `None` then
            a new qml.Device will be initialised.

        :return:
            Returns `self` after training.
        """
        if not dev:
            dev = qml.device(self._device_string, wires=self.wires)

        self._dev = dev

        self._circuit = self._create_circuit()

        self._split_data_for_training(X, y)

        n_batches: int = max(1, len(X) // self._batch_size)

        feature_batches = np.array_split(np.arange(len(self._training_X)), n_batches)

        best_weights: Sequence[float] = self.weights
        best_accuracy: float = self.score(self._validation_X, self._validation_y)

        self.weights = np.array(self.weights, requires_grad=True)
        cost: float
        batch_indices: np.tensor  # Of ints.

        def _batch_cost(weights: Sequence[float]):
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

            if accuracy_validation >= self._accuracy_threshold:
                break

        self.weights = best_weights

        return self

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
        return self._prediction_function(self.get_circuit_expectation_values(features))


class QNNBinaryClassifier(QNNModel, ClassifierMixin):
    """
    This class implements a binary classifier that uses Quantum Neural Networks.

    The classifier expects two classes {0, 1}.
    """

    def __init__(
        self,
        wires: Union[int, Sequence[int]],
        batch_size: int,
        n_epochs: int = 1,
        device_string: str = "lightning.qubit",
        optimizer: Optional[GradientDescentOptimizer] = None,
        embedding_method: Optional[Type[qml.operation.Operation]] = None,
        embedding_kwargs: Optional[Dict[str, Any]] = None,
        layers: Optional[Sequence[Type[qml.operation.Operation]]] = None,
        layers_weights_shapes: Optional[Sequence[Tuple[int, ...]]] = None,
        accuracy_threshold: float = 0.8,
        initial_weights: Optional[Sequence[float]] = None,
        rng_seed: int = 42,
        validation_set_size: float = 0.2,
        prediction_function: Optional[
            Callable[[Sequence[float]], Union[Sequence[float], Sequence[int]]]
        ] = None,
        debug_flag: bool = True,
    ) -> None:
        """
        The constructor for the `QNNBinaryClassifier` class.

        :param wires:
            The wires to use in the VQC or the number of qubits (and wires) used in the
            VQC.
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
            classifier will use `AngleEmbedding`. See `_prepare_default_embedding` for
            parameters details.
        :param embedding_kwargs:
            Keyword arguments for the embedding method. If none are specified the
            classifier will use the default embedding method.
        :param layers:
            Layers to be used in the VQC. The layers will be applied in the given order.
            A double `StronglyEntanglingLayer` will be used if `None` is given.
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
        :param prediction_function:
            A prediction function that will be used to process the output of the VQC.
            If `None` then the default one will be used.
        :param debug_flag:
            A flag informing the classifier if the training info should be printed
            to the console or not.
        """
        super().__init__(
            wires,
            batch_size,
            n_epochs,
            device_string,
            optimizer,
            embedding_method,
            embedding_kwargs,
            layers,
            layers_weights_shapes,
            accuracy_threshold,
            initial_weights,
            rng_seed,
            validation_set_size,
            prediction_function,
            debug_flag,
        )

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
        expectation_values: np.ndarray = np.array(
            [self._circuit(x, weights)[0] for x in X]
        )

        return np.mean((expectation_values - np.array(y)) ** 2)

    def _default_prediction_function(
        self, circuit_outputs: Sequence[Sequence[float]]
    ) -> Sequence[int]:
        """
        The default prediction function of the QNNClassifier.

        :param circuit_output:
            The outputs of the VQC.

        :return:
            Returns classification prediction value for the given problem.
        """
        return [2 * int(val >= 0.0) - 1 for val in [x[0] for x in circuit_outputs]]


class QNNLinearRegression(QNNModel, RegressorMixin):
    """ """

    def __init__(
        self,
        wires: Union[int, Sequence[int]],
        batch_size: int,
        n_epochs: int = 1,
        device_string: str = "lightning.qubit",
        optimizer: Optional[GradientDescentOptimizer] = None,
        embedding_method: Optional[Type[qml.operation.Operation]] = None,
        embedding_kwargs: Optional[Dict[str, Any]] = None,
        layers: Optional[Sequence[Type[qml.operation.Operation]]] = None,
        layers_weights_shapes: Optional[Sequence[Tuple[int, ...]]] = None,
        accuracy_threshold: float = 0.8,
        initial_weights: Optional[Sequence[float]] = None,
        rng_seed: int = 42,
        validation_set_size: float = 0.2,
        prediction_function: Optional[
            Callable[[Sequence[float]], Sequence[float]]
        ] = None,
        debug_flag: bool = True,
    ) -> None:
        """
        The constructor for the `QNNLinearRegression` class.

        :param wires:
            The wires to use in the VQC or the number of qubits (and wires) used in the
            VQC.
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
            A double `StronglyEntanglingLayer` will be used if `None` is given.
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
        :param prediction_function:
            A prediction function that will be used to process the output of the VQC.
            If `None` then the default one will be used.
        :param debug_flag:
            A flag informing the classifier if the training info should be printed
            to the console or not.
        """
        super().__init__(
            wires,
            batch_size,
            n_epochs,
            device_string,
            optimizer,
            embedding_method,
            embedding_kwargs,
            layers,
            layers_weights_shapes,
            accuracy_threshold,
            initial_weights,
            rng_seed,
            validation_set_size,
            prediction_function,
            debug_flag,
        )

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
        expected_values = [self._circuit(x, weights) for x in X]

        predicted_values: np.ndarray = np.array(
            [
                sum([np.log(((i + 1) / 2) / (1 - ((i + 1) / 2))) for i in x])
                for x in expected_values
            ]
        )
        return np.mean(predicted_values - np.array(y) ** 2)

    def _default_prediction_function(
        self, circuit_outputs: Sequence[Sequence[float]]
    ) -> Sequence[float]:
        """
        The default prediction function of the QNNClassifier.

        :param circuit_output:
            The outputs of the VQC.

        :return:
            Returns classification prediction value for the given problem.
        """
        predicted_values: np.ndarray = np.array(
            [
                sum([np.log(((i + 1) / 2) / (1 - ((i + 1) / 2))) for i in x])
                for x in circuit_outputs
            ]
        )

        return [v for v in predicted_values]


class QuantumKernelBinaryClassifier(QMLModel, ClassifierMixin):
    """
    This class implements the binary classifier based on quantum kernels.
    """

    def __init__(
        self,
        wires: Union[int, Sequence[int]],
        n_epochs: int = 10,
        kta_subset_size: int = 5,
        device_string: str = "lightning.qubit",
        optimizer: Optional[GradientDescentOptimizer] = None,
        embedding_method: Optional[Type[qml.operation.Operation]] = None,
        embedding_kwargs: Optional[Dict[str, Any]] = None,
        layers: Optional[Sequence[Type[qml.operation.Operation]]] = None,
        layers_weights_shapes: Optional[Sequence[Tuple[int, ...]]] = None,
        initial_weights: Optional[Sequence[float]] = None,
        rng_seed: int = 42,
        accuracy_threshold: float = 0.8,
        validation_set_size: float = 0.2,
        debug_flag: bool = True,
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
        super().__init__(
            wires,
            device_string,
            optimizer,
            embedding_method,
            embedding_kwargs,
            layers,
            layers_weights_shapes,
            validation_set_size,
            rng_seed,
        )

        self.n_epochs: int = n_epochs

        self._kta_subset_size: int = kta_subset_size

        self._accuracy_threshold = accuracy_threshold
        self._validation_set_size = validation_set_size

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

        self._debug_flag: bool = debug_flag

        self._dev: qml.Device = qml.device(device_string, wires=wires)

        self._classifier: SVC = SVC()

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
            self._embedding_method(features, **self._embedding_kwargs)

            layer_weights = weights[
                start_weights : start_weights + prod(self._layers_weights_shapes[i])
            ]
            start_weights += prod(self._layers_weights_shapes[i])
            layer_weights = np.array(layer_weights).reshape(
                self._layers_weights_shapes[i]
            )
            layer(layer_weights, wires=self.wires)

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
            return [qml.expval(qml.PauliZ((i,))) for i in self.wires]

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
            return qml.probs(wires=self.wires)

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
            A list of classes corresponding to the given lists of features. The classes
            should be from set {-1, 1}.

        :return:
            Returns `self` after training.
        """
        self._dev = qml.device(self._device_string, wires=self.wires)

        kernel: Callable[
            [Sequence[float], Sequence[float], Sequence[float]], float
        ] = self._create_kernel()

        self._split_data_for_training(features_lists, classes)

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
                list(features_lists),
                list(classes),
                lambda x1, x2: kernel(self.weights, x1, x2),
                assume_normalized_kernel=True,
            )

            # Second create a kernel matrix function using the trained kernel.
            def kernel_matrix_function(
                x: Sequence[Sequence[float]], y: Sequence[int]
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
                return qml.kernels.kernel_matrix(
                    list(x),
                    list(y),
                    lambda x1, x2: kernel(self.weights, x1, x2),
                )

            self._classifier = SVC(kernel=kernel_matrix_function).fit(
                features_lists, classes
            )

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


class QuantumClassifier(QMLModel, ClassifierMixin):
    """
    This class implements a quantum classifier based on the multiple binary quantum
    classifiers.
    """

    def __init__(
        self,
        binary_classifiers: Optional[Sequence[QNNBinaryClassifier]],
        wires: int = 2,
        device_string: str = "lightning.qubit",
        accuracy_threshold: float = 0.8,
    ) -> None:

        self._binary_classifiers: Sequence[QNNBinaryClassifier] = binary_classifiers
        self._device_string = device_string
        self.wires = wires
        self.accuracy_threshold = accuracy_threshold

    def fit(
        self,
        features_lists: Sequence[Sequence[float]],
        classes: Sequence[int],
        dev: qml.Device = None,
    ) -> "QuantumClassifier":
        """

        :param features_lists:
        :param classes:

        :note:
            We expect the classes to be natural numbers.
        :return:
        """

        if not dev:
            dev = qml.device(self._device_string, wires=self.wires)

        self._dev = dev

        # Check if there's a binary classifier for each class.
        unique_classes = np.unique(classes)

        if len(unique_classes) > len(self._binary_classifiers):
            raise Exception(
                "Numbers of provided binary classifiers and classes don't match!"
            )

        binary_classifier_accuracy_threshold: float = np.sqrt(self.accuracy_threshold)

        for classifier in self._binary_classifiers:
            classifier._accuracy_threshold = binary_classifier_accuracy_threshold

        # Fit each binary classifier to its respective class.
        for i, binary_classifier in enumerate(self._binary_classifiers):
            classifier_classes = np.array(classes)
            classifier_classes[classifier_classes != unique_classes[i]] = -1
            classifier_classes[classifier_classes == unique_classes[i]] = 1

            binary_classifier.fit(features_lists, classifier_classes, self._dev)

        return self

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
        predictions: List[int] = []

        for x in features:

            max_expectation: float = self._binary_classifiers[
                0
            ].get_circuit_expectation_values([x])[0][0]
            current_class: int = 0

            for i, binary_classifier in enumerate(self._binary_classifiers):
                expectation = binary_classifier.get_circuit_expectation_values([x])[0][
                    0
                ]

                if expectation > max_expectation:
                    max_expectation = expectation
                    current_class = i

            predictions.append(current_class)

        return predictions
