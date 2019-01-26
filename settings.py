from os.path import join

from src.models.pretrained_classifier import PretrainedClassifier

# Program
DEFAULT_TRAINING_PROPORTION = 0.8
DEFAULT_SEED = 46546
DEFAULT_CLASSIFIER = [PretrainedClassifier.XCEPTION]
CLASSIFIER_LIST_SEPERATOR = '_'

# Classifiers
INPUT_SIZE = 299  # The size of the images for learning, is decided by the classifier you use
DEFAULT_POOLING = 'avg'  # What pooling the keras pretrained classifiers uses

# Default file names
RESULTS_GRAPH_FOLDER = "graphs"
VALIDATION_RESULTS_INCORRECT_IMAGE_FOLDER = join('validate', 'incorrectly_labeled')
VALIDATION_RESULTS_CORRECT_IMAGE_FOLDER = join('validate', 'correctly_labeled')

TRAINING_CLASSIFIER_FILE_NAME = 'logreg_classifier.sav'

TRAINING_CLASSIFIER_DATA_FILE_NAME = 'logreg_classifier_data.json'
VALIDATION_DATA_FILE_NAME = 'validation_data.json'

VALIDATION_PREDICTIONS_FILE_NAME = 'validation_predictions.csv'
VALIDATION_PROBABILITIES_FILE_NAME = 'validation_probabilities.csv'
VALIDATION_ANSWERS_FILE_NAME = 'validation_answers.csv'
VALIDATION_METRICS_FILE_NAME = 'validation_metrics.csv'
VALIDATION_MATCHING_DATA_FILE_NAME = 'validation_matching_data.csv'

TRAINING_BREED_DATA_PROPORTIONS_FILE_NAME= 'breed_data_proportions.png'
CORRECT_PREDICTION_DISTRIBUTION_PLOT_FILE_NAME = "correct_prediction_distribution_plot.png"
INCORRECT_PREDICTION_HEATMAP_PLOT_FILE_NAME = "incorrect_prediction_heatmap.png"
INCORRECT_PREDICTION_CLUSTERMAP_PLOT_FILE_NAME = "incorrect_prediction_clustermap.png"
INCORRECT_PREDICTION_DISTRIBUTION_PLOT_FILE_NAME = "incorrect_prediction_distribution_plot.png"
