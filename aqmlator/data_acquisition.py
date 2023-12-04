"""
=============================================================================

    This module contains the functionalities required for receiving the data from
    the user and parsing them into a format that is used throughout the rest of the
    library.

=============================================================================

    Copyright 2023 ACK Cyfronet AGH. All Rights Reserved.

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


import csv
import locale
from abc import ABC, abstractmethod
from dataclasses import dataclass
from os.path import exists
from typing import Any, List, Tuple, Union


@dataclass(init=True, repr=True)
class LearningDatum:
    """
    A general class for holding the user-passed learning data.
    """

    datum_attributes: Tuple[Union[float, str], ...]

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, LearningDatum):
            return tuple(self.datum_attributes) == tuple(other.datum_attributes)
        return NotImplemented


@dataclass(init=True, repr=True)
class SupervisedLearningDatum(LearningDatum):
    """
    A class for holding user-passed data for supervised learning. It holds additional
    information about the class / value of the target function.
    """

    datum_target: Union[float, str, int]

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, SupervisedLearningDatum):
            return (
                tuple(self.datum_attributes) == tuple(other.datum_attributes)
                and self.datum_target == other.datum_target
            )
        return NotImplemented


class DataReceiverInterface(ABC):
    """
    An interface class for all the data receivers.
    """

    @abstractmethod
    def receive_data(self, data_file_path: str) -> List[LearningDatum]:
        """
        The main method of the data receivers. It handles the initial data

        :param data_file_path:
            The path to the file containing the data.
        """
        raise NotImplementedError


class CSVDataReceiver(DataReceiverInterface):
    """
    This class is meant to receive the data given in a CSV format.
    """

    def __init__(
        self, data_separator: str = ",", target_index: Union[None, int] = None
    ) -> None:
        """
        A constructor for the CSVDataReceiver.

        :param data_separator:
            A symbol in the file used to separate the data.
        :param target_index:
            Index of the target value for supervised learning tasks. If set to `None`,
            then the Receiver assumes unsupervised learning.

        """
        self._data_separator: str = data_separator
        self._target_index: Union[int, None] = target_index
        super().__init__()

    def receive_data(self, data_file_path: str) -> List[LearningDatum]:
        """
        The main method of the data receiver. It takes the path to the file containing
        the data and returns a list of LearningDatum objects representing the data in
        the CSV file.

        :param data_file_path:
            The path to the file containing the data.

        :raises FileNotFoundError:
            If the file does not exist.

        :return:
            A list of LearningDatum objects representing the data in the CSV file.
        """
        if not exists(data_file_path):
            raise FileNotFoundError

        data: List[LearningDatum] = []

        with open(
            data_file_path, "r", encoding=locale.getpreferredencoding()
        ) as csv_file:
            reader = csv.reader(
                csv_file,
                "excel",
                delimiter=self._data_separator,
                quoting=csv.QUOTE_NONNUMERIC,
            )

            for row in reader:
                row_data: List[Union[float, int, str]] = list(row)

                if self._target_index is None:
                    # Unsupervised learning
                    data.append(LearningDatum(tuple(row_data)))
                else:
                    # Supervised learning
                    target_value: Union[float, int, str] = row_data.pop(
                        self._target_index
                    )
                    data.append(SupervisedLearningDatum(tuple(row_data), target_value))

        return data
