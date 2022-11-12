__author__ = "Tomasz Rybotycki"


"""
    This script contains tests for the data_acquisition module.
"""

import unittest, csv, dill, os

from typing import Tuple, Union, List

from aqmlator.data_acquisition import (
    SupervisedLearningDatum,
    LearningDatum,
    CSVDataReceiver,
)


class TestDataAcquisition(unittest.TestCase):
    """
    A TestCase class for data_acquisition module.
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

        with open("learning_data.csv", "w") as f:
            writer = csv.writer(f, "excel")
            for data in csv_learning_data:
                writer.writerow(data)

        with open("supervised_learning_data.csv", "w") as f:
            writer = csv.writer(f, "excel")
            for i in range(len(csv_data_targets)):
                writer.writerow(csv_learning_data[i] + [csv_data_targets[i]])

    def tearDown(self) -> None:
        """
        Ensures that the files created during the tests are deleted.
        """
        self._ensure_deleted("learning_datum.dil")
        self._ensure_deleted("learning_data.csv")
        self._ensure_deleted("supervised_learning_datum.dil")
        self._ensure_deleted("supervised_learning_data.csv")

    @staticmethod
    def _ensure_deleted(file_path: str) -> None:
        """
        Ensures that the file on under the given path is deleted.

        :param file_path:
            A path to the file that should be deleted.
        """
        try:
            os.remove(file_path)
        except Exception:
            pass

    def test_learning_datum_equality(self) -> None:
        """
        Tests if equality of learning datum was implemented properly.
        """
        x: Tuple[Union[float, str], ...] = (1.23, "lol")
        datum_1: LearningDatum = LearningDatum(x)
        datum_2: LearningDatum = LearningDatum(x)
        self.assertTrue(
            datum_2 == datum_1, "LearningDatum equality implementation has an error."
        )

    def test_learning_datum_hash(self) -> None:
        """
        Tests if LearningDatum class can be properly serialized.
        """
        x: Tuple[Union[float, str], ...] = (1.23, "lol")
        serialized_object_file: str = "learning_datum.dil"
        datum: LearningDatum = LearningDatum(x)

        with open(serialized_object_file, "wb") as f:
            dill.dump(datum, f)

        with open(serialized_object_file, "rb") as f:
            read_datum: LearningDatum = dill.load(f)

        self.assertTrue(
            datum == read_datum,
            "There's some problem with the LearningDatum serialization.",
        )

    def test_supervised_learning_datum_equality(self) -> None:
        """
        Tests if equality of SupervisedLearningDatum was implemented properly.
        """
        x: Tuple[Union[float, str], ...] = (1.23, "lol")
        y: Union[float, str, int] = 1
        datum_1: SupervisedLearningDatum = SupervisedLearningDatum(x, y)
        datum_2: SupervisedLearningDatum = SupervisedLearningDatum(x, y)
        self.assertTrue(
            datum_2 == datum_1,
            "SupervisedLearningDatum equality implementation has an error.",
        )

    def test_supervised_learning_datum_hash(self) -> None:
        """
        Tests if SupervisedLearningDatum class can be properly serialized.
        """
        x: Tuple[Union[float, str], ...] = (1.23, "lol")
        y: int = 1
        serialized_object_file: str = "supervised_learning_datum.dil"
        datum: SupervisedLearningDatum = SupervisedLearningDatum(x, y)

        with open(serialized_object_file, "wb") as f:
            dill.dump(datum, f)

        with open(serialized_object_file, "rb") as f:
            read_datum: SupervisedLearningDatum = dill.load(f)

        self.assertTrue(
            datum == read_datum,
            "There's some problem with the SupervisedLearningDatum serialization.",
        )

    def test_csv_learning_data_reading(self) -> None:
        """
        Checks if the (unsupervised) learning data is properly read from the file.
        """
        receiver: CSVDataReceiver = CSVDataReceiver()

        data: List[LearningDatum] = receiver.ReceiveData("learning_data.csv")

        self.assertTrue(
            len(data) == len(self._learning_data),
            "Received (unsupervised) learning data don't have proper size.",
        )

        for i in range(len(data)):
            self.assertTrue(
                self._learning_data[i] == data[i], "LearningDatum wasn't read properly."
            )

    def test_csv_supervised_data_reading(self) -> None:
        """
        Checks if the supervised learning data is properly read from the file.
        """
        receiver: CSVDataReceiver = CSVDataReceiver(target_index=3)

        data: List[SupervisedLearningDatum] = receiver.ReceiveData(
            "supervised_learning_data.csv"
        )

        self.assertTrue(
            len(data) == len(self._supervised_learning_data),
            "Received supervised learning data don't have proper size.",
        )

        for i in range(len(data)):
            self.assertTrue(
                self._supervised_learning_data[i] == data[i],
                "SupervisedLearningDatum wasn't read properly.",
            )
