"""
    This module contains the functionalities required for receiving the data from
    the user and parsing them into a format that is used throughout the rest of the
    library.
"""

__author__ = "Tomasz Rybotycki"


import csv
from os.path import exists
from abc import ABC, abstractmethod
from typing import Union, Tuple, List
from dataclasses import dataclass


@dataclass(init=True, repr=True, eq=True)
class LearningDatum:
    """
    A general class for holding the user-passed learning data.
    """

    datum_attributes: Tuple[Union[float, str], ...]


@dataclass(init=True, repr=True, eq=True)
class SupervisedLearningDatum(LearningDatum):
    """
    A class for holding user-passed data for supervised learning. It holds additional
    information about the class / value of the target function.
    """

    datum_target: Union[float, str, int]


class DataReceiverInterface(ABC):
    """
    An interface class for all the data receivers.
    """

    @abstractmethod
    def ReceiveData(self, data_file_path: str) -> List[LearningDatum]:
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

    def ReceiveData(self, data_file_path: str) -> List[LearningDatum]:
        """
        The main method of the data receiver. It takes the

        :param data_file_path:
            The path to the file containing the data.

        :return:
            A list of LearningDatum objects representing the data in the CSV file.
        """
        if not exists(data_file_path):
            raise FileNotFoundError

        data: List[LearningDatum] = []

        with open(data_file_path, "r") as csv_file:
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
