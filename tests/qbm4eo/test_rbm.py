"""
Tests for the rbm.py module.
"""
import io

import dimod

import numpy as np
import torch


from qbm4eo.rbm import qubo_from_rbm_coefficients, RBM


class TestRBMToQUBOConversion:
    """
    Tests for the conversion of an RBM to a QUBO.
    """

    def test_qubo_is_correctly_constructed_for_single_unit_cell_rbm(self) -> None:
        weights = torch.tensor(
            [
                [0.5, 0.4, 0.2, 0.3],
                [0.4, -0.2, 0.01, 1.1],
                [-0.25, 0.5, 0.6, -0.7],
                [1.0, 0.5, -0.25, -0.3],
            ],
            dtype=torch.float64,
        )
        v_bias = torch.tensor([0.1, -0.2, 0.3, 0.4], dtype=torch.float64)
        h_bias = torch.tensor([-0.25, -0.3, 1.0, 0.25], dtype=torch.float64)
        expected_bqm = dimod.BQM(
            {0: 0.1, 1: -0.2, 2: 0.3, 3: 0.4, 4: -0.25, 5: -0.3, 6: 1.0, 7: 0.25},
            {
                (0, 4): 0.5,
                (0, 5): 0.4,
                (0, 6): 0.2,
                (0, 7): 0.3,
                (1, 4): 0.4,
                (1, 5): -0.2,
                (1, 6): 0.01,
                (1, 7): 1.1,
                (2, 4): -0.25,
                (2, 5): 0.5,
                (2, 6): 0.6,
                (2, 7): -0.7,
                (3, 4): 1.0,
                (3, 5): 0.5,
                (3, 6): -0.25,
                (3, 7): -0.3,
            },
            offset=0.0,
            vartype="BINARY",
        )

        bqm = qubo_from_rbm_coefficients(weights, v_bias, h_bias)

        assert bqm == expected_bqm


class TestRBM:
    """
    Tests for the RBM class.
    """

    def test_loading_saved_rbm_gives_back_original_rbm(self) -> None:
        original_rbm: RBM = RBM(15, 13)
        dummy_file: io.BytesIO = io.BytesIO()
        original_rbm.save(dummy_file)
        dummy_file.seek(0)

        loaded_rbm: RBM = RBM.load(dummy_file)

        # Perform the assertions.
        np.testing.assert_array_equal(original_rbm.weights, loaded_rbm.weights)
        np.testing.assert_array_equal(original_rbm.v_bias, loaded_rbm.v_bias)
        np.testing.assert_array_equal(original_rbm.h_bias, loaded_rbm.h_bias)

        assert original_rbm.num_visible == loaded_rbm.num_visible
        assert original_rbm.num_hidden == loaded_rbm.num_hidden
