"""
=============================================================================

    This module contains tests for the functionalities in the data_preprocessing module.

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

from math import isclose
import unittest
from typing import List, Tuple, Union, Sequence
from aqmlator.data_acquisition import LearningDatum, SupervisedLearningDatum
from aqmlator.data_preprocessing import (
    get_targets,
    get_attributes,
    LearningDatumMinMaxScaler,
)


class TestDataPreprocessing(unittest.TestCase):
    """
    A `TestCase` class for data_preprocessing module.
    """

    def setUp(self) -> None:
        """
        Sets up the tests.
        """
        csv_learning_data: List[List[float]] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        csv_data_targets: List[int] = [1, 2]

        self._learning_data: List[LearningDatum] = [
            LearningDatum(tuple(data)) for data in csv_learning_data
        ]

        self._supervised_learning_data: List[SupervisedLearningDatum] = [
            SupervisedLearningDatum(tuple(csv_learning_data[i]), csv_data_targets[i])
            for i in range(len(csv_data_targets))
        ]

        self._scaler_max: float = 3.14
        self._scaler_min: float = 0

        self._scaler: LearningDatumMinMaxScaler = LearningDatumMinMaxScaler(
            (self._scaler_min, self._scaler_max)
        )

    def test_get_targets(self) -> None:
        """
        Tests `get_targets` method.
        """
        targets: List[Union[str, int, float]] = get_targets(
            self._supervised_learning_data
        )

        self.assertTrue(
            len(targets) == len(self._supervised_learning_data),
            "Number of targets and the data is different.",
        )

        for i in range(len(targets)):
            self.assertTrue(
                targets[i] == self._supervised_learning_data[i].datum_target,
                "Targets are ordered differently than expected.",
            )

    def test_get_attributes_on_learning_data(self) -> None:
        """
        Tests `get_attributes` method on `LearningDatum` objects.
        """
        attributes: List[Tuple[Union[str, float], ...]] = get_attributes(
            self._learning_data
        )

        self.assertTrue(
            len(attributes) == len(self._learning_data),
            "Number of attributes and the learning data is different.",
        )

        for i in range(len(attributes)):
            self.assertTrue(
                attributes[i] == self._learning_data[i].datum_attributes,
                "Attributes are ordered differently than expected from learning data.",
            )

    def test_get_attributes_on_supervised_learning_data(self) -> None:
        """
        Tests `get_attributes` method on `SupervisedLearningDatum` objects.
        """

        supervised_attributes: List[Tuple[Union[str, float], ...]] = get_attributes(
            self._supervised_learning_data
        )

        self.assertTrue(
            len(supervised_attributes) == len(self._supervised_learning_data),
            "Number of attributes and the learning data is different.",
        )

        for i in range(len(supervised_attributes)):
            self.assertTrue(
                supervised_attributes[i]
                == self._supervised_learning_data[i].datum_attributes,
                "Attributes are ordered differently than expected from supervised "
                "learning data.",
            )

    def _learning_datum_min_max_scaler_attribute_tests(
        self, data: Sequence[LearningDatum]
    ) -> None:
        """
        Boilerplate code for testing if the datum attributes are fitted correctly.
        """
        fitted_data: Sequence[LearningDatum] = self._scaler.fit_transform(data)

        self.assertTrue(
            len(fitted_data) == len(data),
            "Fitted data has different len than the original data.",
        )

        # Ensure that the original data wasn't transformed in the process.
        for i in range(len(fitted_data)):
            self.assertFalse(
                fitted_data[i] == data[i],
                "Original data was transformed in the process!",
            )

        # Ensure attributes are within specified bounds.
        min_value: float = min(fitted_data[0].datum_attributes)
        max_value: float = max(fitted_data[0].datum_attributes)
        for i in range(1, len(fitted_data)):
            min_value = min(min_value, min(fitted_data[i].datum_attributes))
            max_value = max(max_value, max(fitted_data[i].datum_attributes))

        self.assertTrue(
            isclose(max_value, self._scaler_max),
            f"Max value is not close to the specified max:"
            f" {max_value} != {self._scaler_max}.",
        )
        self.assertTrue(
            isclose(min_value, self._scaler_min),
            f"Min value is not close to the specified min:"
            f" {min_value} != {self._scaler_min}",
        )

    def test_learning_datum_min_max_scaler_on_learning_datum(self) -> None:
        """
        Tests `LearningDatumMinMaxScaler` on `LearningDatum` objects.
        """
        self._learning_datum_min_max_scaler_attribute_tests(self._learning_data)

    def test_learning_datum_min_max_scaler_on_supervised_learning_datum(self) -> None:
        """
        Tests `LearningDatumMinMaxScaler` on `SupervisedLearningDatum` objects.
        """
        self._learning_datum_min_max_scaler_attribute_tests(
            self._supervised_learning_data
        )

        fitted_data: Sequence[LearningDatum] = self._scaler.fit_transform(
            self._supervised_learning_data
        )

        # Check if classes are intact and if the return type is correct.
        for i in range(len(fitted_data)):
            if isinstance(fitted_data[i], SupervisedLearningDatum):
                self.assertTrue(
                    self._supervised_learning_data[i].datum_target
                    == fitted_data[i].datum_target,
                    "Class of the fitted data has changed!",
                )
            else:
                self.assertTrue(False, "Data has wrong type!")
