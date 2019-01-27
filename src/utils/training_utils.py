import re
import random
from os import listdir
from os.path import isdir, join
from typing import List, Tuple

from settings import DEFAULT_SEED

from src.models.image import Image
from src.models.classifier_settings import ImageSource
from src.utils.io_utils import is_image_file
from src.utils.list_utils import extract_unique_values

EXTRACT_PATH_REGEX = re.compile(
    r'^((?P<id>\w+)-(?P<breed>\w+))$',
    re.IGNORECASE)


def load_data_and_split_for_training_validating(image_sources: List[ImageSource],
                                                load_for_breeds: List[str],
                                                training_proportion: float,
                                                seed: int = None) -> Tuple[List[Image], List[Image]]:
    """
    Loads training and validation data for set parameters
    :param load_for_breeds: A list of breads we want to load images from
    :param training_proportion: The proportion of training data we want
    :param seed: A seed value for shuffling the image data, if None it defaults to DEFAULT_SEED
    :return: A tuple with a list of training data and validation data respectively
    """
    all_images = []

    for image_source in image_sources:
        data_from_images = load_all_image_data_for_breeds(image_source, load_for_breeds)
        all_images.extend(data_from_images)

    if training_proportion == 1:
        return all_images, []

    if seed is None:
        random.seed(DEFAULT_SEED)
    else:
        random.seed(seed)

    image_count = len(all_images)
    training_image_count = int(round(training_proportion * image_count))
    random.shuffle(all_images)
    return all_images[:training_image_count], all_images[training_image_count:]


def load_all_image_data_for_breeds(image_source: ImageSource, breeds: List[str]) -> List[Image]:
    """
    Loads all breed images by loading images from each subdirectory separately
    :param image_source: Source for images
    :param breeds: List of breeds to load
    :return: A list of Image objects
    """
    all_images = []
    train_path = image_source.getResourceFolder()
    breed_paths = listdir(train_path)
    for path in breed_paths:
        breed_path = join(train_path, path)
        if isdir(breed_path) and EXTRACT_PATH_REGEX.match(path):
            _, breed_name = split_path_name(path)
            if breed_name in breeds:
                images_in_path = load_images_in_path(breed=breed_name, path=breed_path)
                all_images.extend(images_in_path)
    return all_images


def load_images_in_path(breed: str, path: str) -> List[Image]:
    """
    Takes a path and loads all image files from it
    :param path: A full path to load from
    :return: A list of Image objects
    """
    images = []
    all_files_in_path = listdir(path)
    for file in all_files_in_path:
        image_full_path = join(path, file)
        if is_image_file(image_full_path):
            images.append(
                Image(image_path=image_full_path,
                      image_id=file,
                      image_classification=breed))
    return images


def load_breeds(image_sources: List[ImageSource]) -> List[str]:
    """
    Loads breeds from path with a pre-compiled regex.
    :param image_sources:
    :return: A list of breeds
    """
    breeds = []
    for image_source in [path.getResourceFolder() for path in image_sources]:
        breed_paths = listdir(image_source)
        for path in breed_paths:
            if isdir(join(image_source, path)) and EXTRACT_PATH_REGEX.match(path):
                _, path_breed = split_path_name(path)
                breeds.append(path_breed)
    return extract_unique_values(breeds)


def split_path_name(path_name: str) -> Tuple[str, str]:
    """
    Takes a path name and returns id & breed
    :param path_name: The path name
    :return: id & breed tuple
    """
    matches = EXTRACT_PATH_REGEX.search(path_name)
    path_id = matches.group('id')
    path_breed = matches.group('breed')
    return path_id, path_breed.lower()
