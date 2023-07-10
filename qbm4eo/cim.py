from typing import Any, Dict, Tuple

import dimod
import numpy as np


def vectorize_ising(
    h_map: Dict[int, float], j_map: Dict[Tuple[int, int], float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct dense vector representation of the Ising model.

    :param h_map:
        map i -> h_i. Missing spins are considered to have a bias equal to 0.
    :param j_map:
        map (i, j) -> J_{ij}. If given pair (i, j) is not present, it is assumed that
        J_{ij} = 0. If both (i, j) and (j, i) exist in j_map, the corresponding J_{ij}
        terms are summed for both entries in the output interaction matrix.

    :return:
        a tuple (h_vec, j_mat) with h_vec.shape = (N,), j_mat.shape = (N, N), where N is
        the highest index of spin encountered in h_map and j_map MINUS ONE (see tests
        for examples). h_vec contains linear biases and j_mat contains interactions.
        The j_mat is always symmetric.
    """
    num_spins = (
        max(max(h_map.keys()), max([spin for key in j_map.keys() for spin in key])) + 1
    )

    h_vec = np.zeros(num_spins)
    j_mat = np.zeros((num_spins, num_spins))
    for i, bias in h_map.items():
        h_vec[i] = bias

    for (i, j), coupling in j_map.items():
        j_mat[i, j] = coupling
        j_mat[j, i] = coupling

    return h_vec, j_mat


def ramp(time, tau, alpha, pi, pf):
    return (pf + pi) + (pf - pi) * np.tanh(alpha * (2.0 * time / tau - 1.0))


class CIMSampler(dimod.Sampler):
    def __init__(self, pump, noise):
        self.pump = pump
        self.noise = noise

    def sample_ising(
        self, h, J, initial_state=None, num_reads=1, saturation=1, momentum=0.4, scale=1
    ):
        h_vec, J_mat = vectorize_ising(h, J)
        if initial_state is None:
            initial_state = np.random.uniform(-0.5, 0.5, (num_reads, h_vec.shape[0]))

        all_solutions, all_energies = self._sample_ising_internal(
            h_vec, J_mat, initial_state, num_reads, saturation, momentum, scale
        )

        return dimod.SampleSet.from_samples(list(all_solutions), "SPIN", all_energies)

    def _sample_ising_internal(
        self,
        h_vec,
        J_mat,
        initial_state,
        num_reads=1,
        saturation=1,
        momentum=0.4,
        scale=1,
    ):
        x = initial_state.astype(float).reshape((-1, h_vec.shape[0]))
        if x.shape[0] != num_reads:
            if x.shape[0] != 1:
                raise ValueError(
                    f"Passed {x.shape[0]} initiali states, expected {num_reads} or 1"
                )
            x = np.repeat(x, num_reads, axis=0)
        L = x.shape[1]
        delta_m = np.zeros_like(x)
        for p in self.pump:
            delta_x = (
                p * x
                - scale * (x @ J_mat + h_vec)
                + self.noise.rvs(size=(num_reads, L))
            )
            m = (1.0 - momentum) * delta_x + momentum * delta_m
            x += m * (np.abs(x + m) < saturation)
            delta_m = m
        solution = np.sign(x).astype(int)
        return solution, h_vec @ solution.T + np.diag(solution @ J_mat @ solution.T) / 2

    @property
    def properties(self) -> Dict[str, Any]:
        return {}

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "initial_state": [],
            "num_reads": [],
            "saturation": [],
            "momentum": [],
            "scale": [],
        }
