"""
=============================================================================

    This module contains tests for the classes in tuner.py.

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

import unittest

from typing import Sequence

from aqmlator.tuner import ModelFinder
from sklearn.datasets import make_moons
from numpy.random import RandomState


class TestModelFinder(unittest.TestCase):
    """
    This is a `TestCase` for the `ModelFinder` class.
    """

    def setUp(self) -> None:
        """
        Sets up the tests.

        :Note:
            There's no return value and only a single method to test, so we strive
            to be as minimalistic as possible.
        """

        x: Sequence[Sequence[float]]
        y: Sequence[int]

        x, y = make_moons(
            n_samples=100, shuffle=True, noise=0.1, random_state=RandomState(0),
        )

        n_seeds: int = 2
        n_trials: int = 2

        self.model_finder: ModelFinder = ModelFinder(
            x, y, n_cores=1, n_trials=n_trials, n_seeds=n_seeds
        )

    def test_model_finder_running(self) -> None:
        """
        Tests if `ModelFinder` runs.
        """
        self.model_finder.find_model()
        self.assertTrue(True, "ModelFinder crashed while finding the model!")
