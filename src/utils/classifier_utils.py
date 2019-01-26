from typing import List

from settings import CLASSIFIER_LIST_SEPERATOR
from src.models.pretrained_classifier import PretrainedClassifier


def classifier_list_to_text(classifiers: List[PretrainedClassifier]) -> str:
    """
    Turns a list of classifiers into corresponding string representation separated by a predefined constant
    :param classifiers: List of classifiers
    :return: String representation of classifiers
    """
    return CLASSIFIER_LIST_SEPERATOR.join(classifier.name.lower() for classifier in classifiers)

