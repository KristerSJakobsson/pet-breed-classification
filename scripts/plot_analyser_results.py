#!/usr/bin/python

import sys

from src.execution.validate import plot_validation_graphs_and_images
from src.utils.parameter_utils import parse_parameters

if __name__ == "__main__":
    classifier_details, _ = parse_parameters(sys.argv[1:])
    plot_validation_graphs_and_images(classifier_details)
