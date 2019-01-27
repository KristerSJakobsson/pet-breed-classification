from os import makedirs
from os.path import join, exists, isdir, isfile
from typing import Any
import mimetypes as mt

import jsonpickle
from pandas import DataFrame, read_csv
from sklearn.externals import joblib

from definitions import CLASSIFIERS_PATH


def is_directory(path: str):
    return isdir(path)


def is_file(path: str):
    return isfile(path)


def is_image_file(file_path: str) -> bool:
    if is_file(file_path):
        file_type, image_encoding = mt.guess_type(file_path)
        if file_type is not None and file_type.startswith("image/"):
            return True
        else:
            return False
    else:
        raise FileNotFoundError("The following path is not a file: " + file_path)


def create_directory_if_not_exists(path: str):
    if not exists(path):
        makedirs(path)
    elif is_file(path):
        raise BlockingIOError("Expected path, got file: " + path)


def prepare_storage_get_folder(classifier_name: str = None) -> str:
    """
    Prepares a folder for the specified classifier and returns its path
    :param classifier_name: The name of the classifier
    :return: The path to the created classifier folder
    """
    create_directory_if_not_exists(CLASSIFIERS_PATH)
    if classifier_name is not None:
        classifier_specific_path = join(CLASSIFIERS_PATH, classifier_name)
        create_directory_if_not_exists(classifier_specific_path)
        return classifier_specific_path
    else:
        return CLASSIFIERS_PATH


def get_file_path(filename: str, classifier_name: str = None) -> str:
    """
    Gets a file path for a filename, optionally inside a classifier path
    :param filename: The filename for which we want to get a path
    :param classifier_name: Optionally the classifier for which we want to find the path
    :return: The full path to the file corresponding to the input
    """
    if classifier_name is not None:
        file_path = join(CLASSIFIERS_PATH, classifier_name, filename)
    else:
        file_path = join(CLASSIFIERS_PATH, filename)
    if not exists(file_path):
        raise FileNotFoundError("Could not find file: " + file_path)
    return file_path


def store_dataframe(dataframe: DataFrame, classifier_name: str, filename: str):
    target_folder = prepare_storage_get_folder(classifier_name)
    dataframe.to_csv(join(target_folder, filename), index=True, index_label="id")


def load_dataframe(classifier_name: str, filename: str) -> DataFrame:
    file_path = get_file_path(classifier_name=classifier_name, filename=filename)
    return read_csv(file_path, index_col=0)


def store_sklearn_classifier(sklearn_classifier: Any, classifier_name: str, filename: str):
    target_folder = prepare_storage_get_folder(classifier_name)
    joblib.dump(sklearn_classifier, join(target_folder, filename))


def load_sklearn_classifier(classifier_name: str, filename: str) -> Any:
    file_path = get_file_path(classifier_name=classifier_name, filename=filename)
    return joblib.load(file_path)


def store_serializable_object(serializable_object: Any, filename: str, classifier_name: str = None):
    target_folder = prepare_storage_get_folder(classifier_name)
    json_object = jsonpickle.encode(serializable_object)
    with open(join(target_folder, filename), 'w') as my_file:
        my_file.write(json_object)


def load_serializable_object(filename: str, classifier_name: str = None) -> Any:
    file_path = get_file_path(classifier_name=classifier_name, filename=filename)
    with open(file_path, 'r') as my_file:
        raw_text = my_file.read()
    return jsonpickle.decode(raw_text)


def store_figure(figure: Any, classifier_name: str, image_folder: str, filename: str):
    target_folder = prepare_storage_get_folder(classifier_name)
    image_folder = join(target_folder, image_folder)
    create_directory_if_not_exists(image_folder)
    figure.savefig(join(image_folder, filename))
