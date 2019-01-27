from typing import List

from settings import PARAMETER_LIST_SEPERATOR
from src.models.classifier_settings import PretrainedClassifier, ImageSource


def classifier_type_list_to_text(classifiers: List[PretrainedClassifier]) -> str:
    """
    Turns a list of classifiers into corresponding string representation separated by a predefined constant
    :param classifiers: List of classifiers
    :return: String representation of classifiers
    """
    return PARAMETER_LIST_SEPERATOR.join(classifier.name.lower() for classifier in classifiers)


def classifier_data_source_list_to_text(image_sources: List[ImageSource]) -> str:
    """
    Turns a list of image sources into corresponding string representation separated by a predefined constant
    :param classifiers: List of image source
    :return: String representation of image sources
    """
    return PARAMETER_LIST_SEPERATOR.join(image_source.name.lower() for image_source in image_sources)
