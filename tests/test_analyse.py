# Tests /src/execution/analyse

import unittest

from os.path import join

from src.execution.analyse import analyse_image
from src.models.containers import ClassifierDetails, PretrainedClassifier
from definitions import TEST_DATA_PATH


class TestAnalyse(unittest.TestCase):

    @staticmethod
    def dummy_classifier_details():
        return ClassifierDetails(
            breed_names=["European_hedgehod", "Four-toed_hedgehog", "Hemiechinus", "Long-eared_hedgehog"],
            training_classifier=[PretrainedClassifier.INCEPTION],
            seed=1234,
            training_proportion=1.0
        )

    def test_load_inexisting_file(self):
        dummy_classifier = TestAnalyse.dummy_classifier_details()
        with self.assertRaises(FileNotFoundError):
            analyse_image(image_path=join(TEST_DATA_PATH, "not_exist.png"), classifier_details=dummy_classifier)

    def test_load_inexisting_path(self):
        dummy_classifier = TestAnalyse.dummy_classifier_details()
        with self.assertRaises(FileNotFoundError):
            analyse_image(image_path=join(TEST_DATA_PATH, "not_exist"), classifier_details=dummy_classifier)

    def test_load_non_image_file(self):
        dummy_classifier = TestAnalyse.dummy_classifier_details()
        file_path = join(TEST_DATA_PATH, "dummyfile.txt")
        with self.assertRaises(FileNotFoundError):
            analyse_image(image_path=file_path, classifier_details=dummy_classifier)
