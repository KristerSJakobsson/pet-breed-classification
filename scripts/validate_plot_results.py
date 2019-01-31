#!/usr/bin/python

import sys

from src.execution.validate import plot_validation_result_graph_and_tagged_images
from src.utils.parameter_utils import parse_parameters

if __name__ == "__main__":
    classifier_details, _ = parse_parameters(sys.argv[1:])
    plot_validation_result_graph_and_tagged_images(classifier_details)
