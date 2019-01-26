#!/usr/bin/python

import sys

from src.execution.validate import validate_classifier
from src.utils.parameter_utils import parse_parameters

if __name__ == "__main__":
    classifier_details, _ = parse_parameters(sys.argv[1:])
    validate_classifier(classifier_details)
