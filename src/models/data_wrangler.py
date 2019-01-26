import random
from typing import List, Tuple, TypeVar

import pandas as pd
from numpy import ndarray

from settings import DEFAULT_SEED, INPUT_SIZE
from src.utils.image_utils import load_and_preprocess_multiple_images

from .image import Image
from src.utils.list_utils import extract_unique_values

T = TypeVar('T')


class DataWrangler:
    def __init__(self, image_files: List[Image]):
        """
        Wrangles data for use
        :param image_files: List of Image objects
        """
        # Store default values for variables
        self.__image_input_size = INPUT_SIZE
        self.__seed = DEFAULT_SEED

        # Store a shuffled list of pictures
        random.seed(self.__seed)
        self._image_files = image_files

        # Default to none
        self.x_data = None
        self.y_data = None
        self.file_labels = None

    @property
    def image_input_size(self):
        return self.__image_input_size

    @image_input_size.setter
    def image_input_size(self, value: int):
        if value < 0:
            raise ValueError("Image size must be greater than 0")
        self.__image_input_size = value

    def _load_and_preprocess_data(self, file_paths: List[str]) -> ndarray:
        """
        Loads data and preprocess it for use with classifiers
        :param file_paths: The paths for the files to load
        :return: The preprocessed matrix data
        """
        image_size = (self.__image_input_size, self.__image_input_size)
        return load_and_preprocess_multiple_images(image_paths=file_paths,
                                                   image_size=image_size)

    @property
    def seed(self):
        return self.__seed

    @seed.setter
    def seed(self, value: int):
        self.__seed = value
        random.seed(self.__seed)

    def _load_training_data(self, breeds_list: List[str]) -> Tuple[ndarray, ndarray, List[str]]:
        file_ids = []
        file_classifications = []
        file_paths = []

        for image in self._image_files:
            file_ids.append(image.image_id)
            file_classifications.append(image.image_classification)
            file_paths.append(image.image_path)

        # The get_dummies function enforces some prefix for the column names...
        table_column_prefix = "breed"

        dataframe_classification = pd.DataFrame(index=file_ids, data={'breed': file_classifications})
        classification_pivot = pd.get_dummies(dataframe_classification, prefix=table_column_prefix)

        # Fix the order to match breed list
        column_order = [table_column_prefix + "_" + breed for breed in breeds_list]
        classification_pivot = classification_pivot.reindex(labels=column_order, axis=1)

        x_training_data = self._load_and_preprocess_data(file_paths)
        y_training_data = classification_pivot.values
        return x_training_data, y_training_data, file_ids

    def execute_load_training_data(self, shuffle: bool, breeds_list: List[str]):
        if shuffle:
            # Shuffle resources before training
            random.shuffle(self._image_files)

        # Load resources files (in shuffled order)
        x_train, y_train, labels = self._load_training_data(breeds_list)

        self.x_data = x_train
        self.y_data = y_train
        self.file_labels = labels
