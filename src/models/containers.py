from typing import List

from numpy import ndarray, transpose
from pandas import DataFrame
from settings import PARAMETER_LIST_SEPERATOR

from src.utils.classifier_setting_utils import classifier_type_list_to_text
from .classifier_settings import PretrainedClassifier, ImageSource


# Warning: This class inherits from object, if you do not specify this jsonpickle might fail
class ClassifierDetails(object):
    def __init__(self, breed_names: List[str], training_classifier: List[PretrainedClassifier], seed: int,
                 training_proportion: float, image_sources: List[ImageSource]):
        """
        Details of a classifier
        :param breed_names: List of breed names
        :param training_classifier: List of training classifiers
        :param seed: A seed to use throughout the program
        :param training_proportion: Proportion of data to use in training (remaining is used forvalidation)
        :param image_sources: List of image sources (to load image data from)
        """
        self.training_breed_names = breed_names
        self.training_classifiers = training_classifier
        self.seed = seed
        self.training_proportion = training_proportion
        self.image_sources = image_sources

    def get_name(self) -> str:
        return classifier_type_list_to_text(
            self.training_classifiers) + PARAMETER_LIST_SEPERATOR + "classifier" + PARAMETER_LIST_SEPERATOR + str(
            self.seed) + PARAMETER_LIST_SEPERATOR + str(self.training_proportion)


class ClassifierResult(object):
    def __init__(self, probability: ndarray, prediction: ndarray, image_list: List[str],
                 classifier_details: ClassifierDetails):
        self._prediction = prediction
        self._probability = probability
        self._classifier_details = classifier_details
        self._image_list = image_list

    @property
    def classifier_details(self) -> ClassifierDetails:
        return self._classifier_details

    @property
    def prediction_ndarray(self) -> ndarray:
        return self._prediction

    @property
    def probability_ndarray(self) -> ndarray:
        return self._probability

    @property
    def image_list(self) -> List[str]:
        return self._image_list

    def _get_breed_list(self):
        return self._classifier_details.training_breed_names

    def get_probability(self) -> DataFrame:
        breed_list = self._get_breed_list()
        image_list = self._image_list

        probability_results = DataFrame(data=transpose(self._probability), columns=image_list, index=breed_list)
        probability_results.rename_axis("breeds")

        return probability_results

    def get_prediction(self) -> DataFrame:
        breed_list = self._get_breed_list()
        image_list = self._image_list

        selected_breed = []
        selected_probability = []
        for index, value in enumerate(self._prediction):
            breed_value = value.astype(int)
            selected_breed.append(breed_list[breed_value])
            selected_probability.append(self._probability[index, breed_value])

        selection_results = DataFrame(data=transpose([selected_breed, selected_probability]),
                                      columns=["prediction", "probability"],
                                      index=image_list)
        selection_results.rename_axis("filename")
        return selection_results
