from src.models.pretrained_classifier import PretrainedClassifier

# Program
DEFAULT_TRAINING_PROPORTION = 1  # Sets what proportion is used for training (the rest is used for validation)
DEFAULT_SEED = 63634 # The seed for randomizing resources throughout the program (for example when randomly picking resources for training/validating)
DEFAULT_CLASSIFIER = [PretrainedClassifier.INCEPTION, PretrainedClassifier.XCEPTION]
CLASSIFIER_LIST_SEPERATOR = '_'

# Classifiers
INPUT_SIZE = 299  # The size of the images for learning, is decided by the classifier you use
POOLING = 'avg'  # What pooling the keras pretrained classifiers uses
CLASSIFIER_NAME = 'inception_xception_classifier_' + str(DEFAULT_SEED) + '_' + str(DEFAULT_TRAINING_PROPORTION)

# Default file names
LEARNER_CLASSIFIER_FILE_NAME = 'logreg_learner_classifier.sav'

LEARNER_DATA_FILE_NAME = 'logreg_learner_data.json'
VALIDATION_DATA_FILE_NAME = 'validation_data.json'

VALIDATION_PREDICTIONS_FILE_NAME = 'validation_predictions.csv'
VALIDATION_PROBABILITIES_FILE_NAME = 'validation_probabilities.csv'
VALIDATION_ANSWERS_FILE_NAME = 'validation_answers.csv'
VALIDATION_METRICS_FILE_NAME = 'validation_metrics.csv'
VALIDATION_MATCHING_DATA_FILE_NAME = 'validation_matching_data.csv'

VALIDATION_CORRECT_PREDICTION_DISTRIBUTION_PLOT_FILE_NAME = "correct_prediction_distribution_plot.png"
VALIDATION_INCORRECT_PREDICTION_HEATMAP_PLOT_FILE_NAME = "incorrect_prediction_heatmap.png"
VALIDATION_INCORRECT_PREDICTION_CLUSTERMAP_PLOT_FILE_NAME = "incorrect_prediction_clustermap.png"
VALIDATION_INCORRECT_PREDICTION_DISTRIBUTION_PLOT_FILE_NAME = "incorrect_prediction_distribution_plot.png"
