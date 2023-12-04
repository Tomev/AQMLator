"""
=============================================================================

This file is a part of qbm4eo project.

https://github.com/FeralQubits/qbm4eo

=============================================================================

It has been modified as a part of the EuroHPC PL project funded at the Smart Growth
Operational Programme 2014-2020, Measure 4.2 under the grant agreement no.
POIR.04.02.00-00-D014/20-00.

=============================================================================
"""
import abc
import io
from itertools import islice
from typing import Any, Callable, Dict, Generator, Optional, Tuple, Union

import dimod
import numpy as np
from dimod.core.sampler import Sampler
from numpy.typing import NDArray
from scipy.special import expit
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

INITIAL_COEFFICIENT_SCALE: float = 0.1


def infinite_dataloader_generator(
    data_loader: Union[
        DataLoader[Tuple[Any, Any]],
        Generator[Tuple[Any, Any], Any, Any],
    ]
) -> Generator[
    Tuple[Any, Any], None, None
]:  # https://www.linuxjournal.com/content/pythons-mypy-callables-and-generators
    """
    A generator that infinitely yields data batches from the given dataloader.

    :param data_loader:
        A PyTorch DataLoader initialized with the desired dataset.

    :return:
        Yields an index of a batch and the data from the data_loader.
    """
    while True:
        for batch_idx, (data, target) in enumerate(data_loader):
            yield batch_idx, (data, target)


def qubo_from_rbm_coefficients(
    weights: Union[Tensor, NDArray[np.float32]],
    v_bias: Union[Tensor, NDArray[np.float32]],
    h_bias: Union[Tensor, NDArray[np.float32]],
) -> dimod.BQM:
    """Create a QUBO problem representing RBM with given coefficients.

    :param weights:
        A square interaction matrix.
    :param v_bias:
        An N-element visible layer bias vector.
    :param h_bias:
        An M-element hidden layer bias vector.

    :return:
        A QUBO, represented as dimod.BQM with N+M variables, such that:
            -   variables 0,...,N-1 correspond to hidden layer
            -   variables 0,...,M-1 correspond to visible layer
            -   visible layer biases correspond to linear coefficients of first N
                variables
            -   hidden layer biases correspond to linear coefficients of second M
                variables
            -   weights correspond to interaction terms between first N and second M
                variables.

    .. note::
        This function does not allow for manipulating how RBM variables are mapped to
        the QUBO variables. This is not a problem if QUBO is to be used with an
        unstructured sampler. For sampler, the intended usage is to wrap it in the
        EmbeddingComposite.
    """
    linear: Dict[int, float] = {
        **{i: float(bias) for i, bias in enumerate(v_bias)},
        **{i: float(bias) for i, bias in enumerate(h_bias, start=len(v_bias))},
    }

    quadratic: Dict[Tuple[int, int], float] = {
        (i, j + len(v_bias)): float(weights[i, j])
        for i in range(len(v_bias))
        for j in range(len(h_bias))
    }

    return dimod.BQM(linear, quadratic, offset=0, vartype="BINARY")


class RBM:
    """
    A class implementing the Restricted Boltzmann Machine.
    """

    def __init__(
        self,
        num_visible: int,
        num_hidden: int,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """
        A default constructor for the RMB class instances.

        :param num_visible:
            Number of neurons in the visible layer.
        :param num_hidden:
            Number of neurons in the hidden layer.
        :param rng:
            Random number generator to be used by the RBM.
        """
        self.num_visible: int = num_visible
        self.num_hidden: int = num_hidden
        self.rng: np.random.Generator = (
            rng if rng is not None else np.random.default_rng()
        )

        self.weights: NDArray[np.float32] = (
            self.rng.normal(size=(self.num_visible, self.num_hidden))
            * INITIAL_COEFFICIENT_SCALE
        )

        self.v_bias: NDArray[np.float32] = (
            self.rng.normal(size=self.num_visible) * INITIAL_COEFFICIENT_SCALE
        )

        self.h_bias: NDArray[np.float32] = (
            self.rng.normal(size=self.num_hidden) * INITIAL_COEFFICIENT_SCALE
        )

    def h_probs_given_v(self, v_batch: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Compute the probabilities of the hidden layer neurons being active given the
        visible layer neurons' states.

        :param v_batch:
            A batch of visible layer neurons' states.

        :return:
            A batch of probabilities of the hidden layer neurons being active.
        """
        return expit(self.h_bias + v_batch @ self.weights)

    def sample_h_given_v(self, v_batch: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Sample the hidden layer neurons' states given the visible layer neurons' states.

        :param v_batch:
            A batch of visible layer neurons' states.

        :return:
            A batch of hidden layer neurons' states.
        """
        probs: NDArray[np.float32] = self.h_probs_given_v(v_batch)

        return (self.rng.random(probs.shape) < probs).astype(float)

    def v_probs_given_h(self, h_batch: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Compute the probabilities of the visible layer neurons' states given the hidden
        layer neurons' states.

        :param h_batch:
            A batch of hidden layer neurons' states.

        :return:
            A batch of probabilities of the visible layer neurons' states.
        """
        return expit(self.v_bias + h_batch @ self.weights.T)

    def sample_v_given_h(self, h_batch: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Sample the visible layer neurons' states given the hidden layer neurons' states.

        :param h_batch:
            A batch of hidden layer neurons' states.

        :return:
            A batch of visible layer neurons' states.
        """
        probs: NDArray[np.float32] = self.v_probs_given_h(h_batch)
        return (self.rng.random(probs.shape) < probs).astype(float)

    def reconstruct(self, v_batch: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Reconstruct the visible layer neurons' states given the visible layer neurons'
        states.

        :param v_batch:
            A batch of visible layer neurons' states.

        :return:
            A batch of reconstructed visible layer neurons' states.
        """
        return self.v_probs_given_h(self.h_probs_given_v(v_batch))

    def save(self, file: Union[str, io.BytesIO]) -> None:
        """
        Save the RBM to a file.

        :param file:
            A file to save the RBM to.
        """
        np.savez(file, v_bias=self.v_bias, h_bias=self.h_bias, weights=self.weights)

    @classmethod
    def load(cls, file: Union[str, io.BytesIO]) -> "RBM":
        """
        Load the RBM from a file.

        :param file:
            A file to load the RBM from.

        :return:
            A loaded RBM.
        """
        data: Dict[str, NDArray[np.float32]] = np.load(file)

        weights: NDArray[np.float32] = data["weights"]
        v_bias: NDArray[np.float32] = data["v_bias"]
        h_bias: NDArray[np.float32] = data["h_bias"]

        rbm: "RBM" = cls(num_visible=len(v_bias), num_hidden=len(h_bias))
        rbm.weights = weights
        rbm.v_bias = v_bias
        rbm.h_bias = h_bias
        return rbm


class RBMTrainer:
    """
    A base class for implementing the RBM training algorithms.

    TODO TR:    There are some nasty casting between np.matrix and np.ndarray that
                is there to ensure proper transposition or multiplications when one
                of the matrix dimension is 1. This probable could be done more
                elegantly.
    """

    def __init__(self, num_steps: int) -> None:
        """
        A default constructor for the RBMTrainer class instances.

        :param num_steps:
            Number of training steps.
        """
        self.num_steps: int = num_steps

    def fit(
        self,
        rbm: RBM,
        data_loader: Union[
            DataLoader[Tuple[Any, Any]],
            Generator[Tuple[Any, Any], Any, Any],
        ],
        callback: Callable[[int, RBM, float], None] = None,
    ) -> None:
        """
        Fits the RBM to the data.

        :param rbm:
            An RBM to be trained.
        :param data_loader:
            A data loader.
        :param callback:
            A callback function to be called after each training step.
        """
        for i, (_, (batch, _)) in enumerate(
            pbar := tqdm(
                islice(infinite_dataloader_generator(data_loader), self.num_steps),
                total=self.num_steps,
            )
        ):
            batch = np.matrix(batch.detach().cpu().numpy().squeeze())
            self.training_step(rbm, batch)
            loss: float = (
                (np.array(batch - rbm.reconstruct(batch)) ** 2).sum()
                / batch.shape[0]
                / batch.shape[1]
            )

            pbar.set_postfix(loss=loss)

            if callback is not None:
                callback(i, rbm, loss)

    @abc.abstractmethod
    def training_step(self, rbm: RBM, batch: NDArray[np.float32]) -> None:
        """
        A single training step.

        :param rbm:
            An RBM to be trained.
        :param batch:
            A batch of data.
        """
        pass


class AnnealingRBMTrainer(RBMTrainer):
    """
    A class for training the RBM using the annealing samplers (D-Wave of CIMSampler).
    """

    def __init__(
        self,
        num_steps: int,
        sampler: Sampler,
        qubo_scale: float = 1.0,
        learning_rate: float = 0.01,
        **sampler_kwargs: Dict[str, Any],
    ) -> None:
        """
        A default constructor for the AnnealingRBMTrainer class instances.

        :param num_steps:
            Number of training steps.
        :param sampler:
            A sampler to be used during the training.
        :param qubo_scale:
            A scaling factor for the QUBO.
        :param learning_rate:
            A learning rate.
        :param sampler_kwargs:
            Additional keyword arguments to be passed to the sampler.
        """
        super().__init__(num_steps)
        self.sampler: Sampler = sampler
        self.sampler_kwargs: Dict[str, Any] = sampler_kwargs
        self.qubo_scale: float = qubo_scale
        self.learning_rate: float = learning_rate

    def training_step(self, rbm: RBM, batch: NDArray[np.float32]) -> None:
        """
        A single training step.

        :param rbm:
            An RBM to be trained.
        :param batch:
            The RBMs' visible layer neurons' states.
        """
        # Conditional probabilities given visible batch input
        hidden: NDArray[np.float32] = rbm.h_probs_given_v(batch)
        # Construct QUBO from this RBM
        bqm: dimod.BQM = qubo_from_rbm_coefficients(rbm.weights, rbm.v_bias, rbm.h_bias)
        # Scaling to compensate the temperature difference. Strangely, it seems
        # that in dimod this operation has to be done in place.
        bqm.scale(self.qubo_scale)
        # Take a sample of the same size as batch, extract only visible and hidden
        # variables. If the sampler supports num_reads, use it, otherwise repeat
        # the sampling for each data point in the batch.
        if "num_reads" in self.sampler.parameters:
            sample = self.sampler.sample(
                bqm, num_reads=len(batch), **self.sampler_kwargs
            ).record["sample"]
        else:
            sample = dimod.concatenate(
                [
                    self.sampler.sample(bqm, **self.sampler_kwargs)
                    for _ in range(len(batch))
                ]
            ).record["sample"]
        # Split, remembering that first variables correspond to hidden layer.
        sample_v = sample[:, : rbm.num_visible]
        sample_h = sample[:, rbm.num_visible :]
        # Update weights.
        rbm.weights += (
            self.learning_rate * (batch.T @ hidden - sample_v.T @ sample_h) / len(batch)
        )
        # And biases
        a = np.asarray(batch - sample_v).sum(axis=0)

        print(type(a))

        rbm.v_bias += (
            self.learning_rate * np.asarray(batch - sample_v).sum(axis=0).squeeze()
        )
        rbm.h_bias += (
            self.learning_rate * np.asarray(hidden - sample_h).sum(axis=0).squeeze()
        )


class CD1Trainer(RBMTrainer):
    """
    A class for training the RBM using the Contrastive Divergence 1 algorithm.
    """

    def __init__(self, num_steps: int, learning_rate: float = 0.01):
        """
        A default constructor for the CD1Trainer class instances.

        :param num_steps:
            Number of training steps.
        :param learning_rate:
            A learning rate.
        """
        super().__init__(num_steps)
        self.learning_rate = learning_rate

    def training_step(self, rbm: RBM, batch: NDArray[np.float32]) -> None:
        """
        A single training step.

        :param rbm:
            An RBM to be trained.
        :param batch:
            The RBMs' visible layer neurons' states.
        """
        # Conditional probabilities given visible batch input
        hidden_1: NDArray[np.float32] = np.matrix(rbm.h_probs_given_v(batch))

        # Propagate hidden -> visible -> hidden again
        visible_2: NDArray[np.float32] = rbm.v_probs_given_h(hidden_1)
        hidden_2: NDArray[np.float32] = np.matrix(rbm.h_probs_given_v(visible_2))

        # Update weights
        rbm.weights += np.array(
            self.learning_rate
            * (np.matrix(batch).T @ hidden_1 - np.matrix(visible_2).T @ hidden_2)
            / len(batch)
        )
        # And biases
        rbm.v_bias += self.learning_rate * np.array(batch - visible_2).sum(axis=0)
        rbm.h_bias += self.learning_rate * np.array(hidden_1 - hidden_2).sum(axis=0)
