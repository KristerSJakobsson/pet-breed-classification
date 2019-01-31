#!/usr/bin/python

import sys

from src.execution.analyse import analyse_image
from src.utils.parameter_utils import parse_parameters


def usage_arguments():
    return """
    Argument: Path to a file or folder. 
    """


if __name__ == "__main__":
    classifier_details, args = parse_parameters(sys.argv[1:], usage_arguments())
    path = args[0]
    analyse_image(path, classifier_details)
