#!/usr/bin/python

import sys
from getopt import getopt, GetoptError
from typing import List, Tuple, Any

from settings import DEFAULT_SEED, DEFAULT_TRAINING_PROPORTION, DEFAULT_CLASSIFIER, CLASSIFIER_LIST_SEPERATOR
from src.models.pretrained_classifier import PretrainedClassifier
from src.models.containers import ClassifierDetails
from src.utils.training_utils import load_breeds
from src.utils.classifier_utils import classifier_list_to_text


def usage_shared_parameters():
    """
    Gives the explanation text for the parameters
    :return: Explanation text
    """
    return """
    Parameters:
    -h --help: 
    Show help this help text.
    -c --classifier: 
    Set classifiers to use.
    Can be either of: inception xception inception""" + CLASSIFIER_LIST_SEPERATOR + """xception
    Default: """ + classifier_list_to_text(DEFAULT_CLASSIFIER) + """
    -s --seed: 
    Set seed, can be any integer number
    Default: """ + str(DEFAULT_SEED) + """
    -t --training_proportion: 
    Set what proportion of the image data set is used for training, and what is used for validation.
    Can be a value between 0 and 1, however, if it is 1 then validation for the classifier will be disabled.
    Default: """ + str(DEFAULT_TRAINING_PROPORTION) + """
    """


def validate_pretrained_classifier_input(input: str) -> List[PretrainedClassifier]:
    """
    Validates the pretrained classifier input
    :param input: The pretrained classifier before parsing
    :return: The pretrained classifier after parsing
    """
    pretrained_classifier = []
    for classifier in input.split(CLASSIFIER_LIST_SEPERATOR):
        if classifier.lower() == PretrainedClassifier.INCEPTION.name.lower():
            pretrained_classifier.append(PretrainedClassifier.INCEPTION)
        elif classifier.lower() == PretrainedClassifier.XCEPTION.name.lower():
            pretrained_classifier.append(PretrainedClassifier.XCEPTION)
        else:
            raise ValueError("Invalid classifier parameter input " + classifier)
    return pretrained_classifier


def validate_training_proportion_input(input: str) -> float:
    """
    Validates the training proportion input
    :param input: The training proportion before parsing
    :return: The training proportion after parsing
    """
    training_proportion = float(input)
    if training_proportion > 1 or training_proportion <= 0:
        raise ValueError(
            "Invalid training proportion input. Must be strictly greater than 0 and less than or equal to 1")
    return training_proportion


def validate_seed_input(input: str) -> int:
    """
    Validates the seed input
    :param input: The seed value before parsing
    :return: The seed value after parsing
    """
    return int(input)


def parse_parameters(argv: List[str], argument_explanation: str = None) -> Tuple[ClassifierDetails, List[str]]:
    """
    Parse parameters when executing script
    :param argv: A list of input arguments specified
    :param argument_explanation: A text explaining the arguments
    :return: A tuple with the ClassifierDetails and specified arguments
    """
    def display_usage_text():
        if argument_explanation is None:
            print(usage_shared_parameters())
        else:
            print(argument_explanation + usage_shared_parameters())

    try:
        options, arguments = getopt(argv, "hc:s:t:", ["help", "classifiers=", "seed=", "training_proportion="])
    except GetoptError as err:
        print(err)
        display_usage_text()
        sys.exit(2)

    classifier_details = parse_options_for_classifier(options, display_usage_text)

    return classifier_details, arguments


def parse_options_for_classifier(options: List[Any], display_usage_text: Any) -> ClassifierDetails:
    """
    Takes option for the executed script
    :param options: A list of options specified at execution
    :param display_usage_text: A callback function that prints the usage text
    :return: A ClassifierDetails object representing the options specified (or default if not specified)
    """
    classifiers = DEFAULT_CLASSIFIER
    seed = DEFAULT_SEED
    training_proportion = DEFAULT_TRAINING_PROPORTION

    # TODO: Breeds could also be option input
    breed_names = load_breeds()

    try:
        for option, value in options:
            if option in ("-h", "--help"):
                display_usage_text()
                sys.exit(2)
            elif option in ("-c", "--classifiers"):
                classifiers = validate_pretrained_classifier_input(value)
            elif option in ("-s", "--seed"):
                seed = validate_seed_input(value)
            elif option in ("-t", "--training_proportion"):
                training_proportion = validate_training_proportion_input(value)
            else:
                raise RuntimeError("Invalid option " + option)
    except Exception as err:
        print(err)
        display_usage_text()
        sys.exit(2)

    return ClassifierDetails(breed_names=breed_names,
                             training_classifier=classifiers,
                             seed=seed,
                             training_proportion=training_proportion)
