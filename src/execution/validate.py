import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import log_loss, accuracy_score

from settings import *

from src.models.data_wrangler import DataWrangler
from src.models.logreg_classifier import ExistingClassifier
from src.models.containers import ClassifierDetails
from src.utils.image_utils import create_image_figure
from src.utils.io_utils import store_dataframe, load_dataframe, load_serializable_object, store_figure


def validate_classifier(classifier_details: ClassifierDetails):
    breeds_list = classifier_details.training_breed_names
    classifier_name = classifier_details.get_name()

    # Load validation resources in classifier
    validation_data = load_serializable_object(filename=VALIDATION_DATA_FILE_NAME, classifier_name=classifier_name)
    if len(validation_data) <= 0:
        raise ValueError(
            "Tried to execute validation on a classifier without validation resources.")

    data_wrangler = DataWrangler(validation_data)
    data_wrangler.image_input_size = INPUT_SIZE

    machine_learner = ExistingClassifier(classifier_name=classifier_name)
    machine_learner.classifier_name = classifier_name
    data_wrangler.execute_load_training_data(False, breeds_list=breeds_list)

    x_validation_data = data_wrangler.x_data
    y_validation_data = data_wrangler.y_data
    validation_labels = data_wrangler.file_labels

    classifier_result = machine_learner.apply_to_stored_learner(x_validation_data, validation_labels)
    image_list = classifier_result.image_list

    # ------------------- Store validation predictions and probabilities ---------------------
    predictions = classifier_result.get_prediction()

    store_dataframe(predictions, classifier_name, VALIDATION_PREDICTIONS_FILE_NAME)

    probabilities = classifier_result.get_probability()

    store_dataframe(probabilities, classifier_name, VALIDATION_PROBABILITIES_FILE_NAME)

    # ------------------- Store correct labels ---------------------
    correct_breeds = (y_validation_data * range(y_validation_data.shape[1])).sum(axis=1)

    answers = pd.DataFrame(data=[breeds_list[i.astype(int)] for i in correct_breeds],
                           index=image_list,
                           columns=['answer'])

    store_dataframe(answers, classifier_name, VALIDATION_ANSWERS_FILE_NAME)

    # ------------------- Store metrics ---------------------
    data_size = int(len(validation_data))
    logarithmic_loss = log_loss(y_validation_data, classifier_result.probability_ndarray)
    score = accuracy_score((y_validation_data * range(len(breeds_list))).sum(axis=1), classifier_result.prediction_ndarray)

    metrics = pd.DataFrame(data=[data_size, logarithmic_loss, score], columns=["value"],
                           index=["Data Records", "Logarithmic Loss", "Accuracy Score"])

    store_dataframe(metrics, classifier_name, VALIDATION_METRICS_FILE_NAME)


def plot_validation_graphs_and_images(classifier_details: ClassifierDetails):
    classifier_name = classifier_details.get_name()

    # Load validation resources in classifier
    validation_data = load_serializable_object(filename=VALIDATION_DATA_FILE_NAME, classifier_name=classifier_name)
    if len(validation_data) <= 0:
        raise ValueError(
            "Tried to execute validation on a classifier without validation resources.")
    validation_dict = {item.image_id: item.image_path for item in validation_data}

    predictions = load_dataframe(classifier_name, VALIDATION_PREDICTIONS_FILE_NAME)
    answers = load_dataframe(classifier_name, VALIDATION_ANSWERS_FILE_NAME)
    classifier_details = load_serializable_object(classifier_name=classifier_name, filename=LEARNER_DATA_FILE_NAME)

    predictions_and_answers = pd.merge(left=answers, right=predictions, left_index=True, right_index=True)
    breeds = classifier_details.training_breed_names
    zero_matrix = np.zeros((len(breeds), len(breeds)), dtype=int)

    heat_matrix = pd.DataFrame(data=zero_matrix, index=breeds, columns=breeds)

    def add_one_to_heat_matrix(x, y):
        heat_matrix.at[x, y] = heat_matrix.at[x, y] + 1

    correct_prediction_indexes = []
    incorrect_prediction_indexes = []
    for index, row in predictions_and_answers.iterrows():
        predicted_breed = row['prediction']
        actual_breed = row['answer']
        add_one_to_heat_matrix(actual_breed, predicted_breed)
        if predicted_breed == actual_breed:
            correct_prediction_indexes.append(index)
        else:
            incorrect_prediction_indexes.append(index)

    store_dataframe(heat_matrix, classifier_name, VALIDATION_MATCHING_DATA_FILE_NAME)

    # Set diagonal to 0
    heat_matrix.values[[np.arange(len(breeds))] * 2] = 0

    # Path where we save heatmap & clustermap
    maps_file_path = "validate_maps"

    # Heatmap
    plt.subplots(figsize=(20, 20))
    heatmap = sns.heatmap(heat_matrix, cbar=True, center=0, cmap="vlag", fmt="d", linewidths=.75, annot=True)
    heatmap.set(xlabel='Predicted', ylabel='Actual')
    store_figure(heatmap.get_figure(), classifier_name, maps_file_path,
                 VALIDATION_INCORRECT_PREDICTION_HEATMAP_PLOT_FILE_NAME)
    plt.show()

    # Remove rows and columns with only 0 values (no mismatch)
    heat_matrix = heat_matrix.loc[(heat_matrix != 0).any(axis=1)]
    heat_matrix = heat_matrix.loc[:, (heat_matrix != 0).any(axis=0)]

    clustermap = sns.clustermap(heat_matrix, center=0, cmap="vlag",
                                linewidths=.75, row_cluster=True, col_cluster=True, figsize=(20, 20))
    store_figure(clustermap, classifier_name, maps_file_path, VALIDATION_INCORRECT_PREDICTION_CLUSTERMAP_PLOT_FILE_NAME)
    plt.show()

    incorrect_image_file_path = "validate_incorrectly_labeled"
    correct_image_file_path = "validate_correctly_labeled"

    incorrect_prediction_probabilities = []
    correct_prediction_probabilities = []

    for index, row in predictions_and_answers.iterrows():
        image_id = index
        predicted_breed = row['prediction']
        actual_breed = row['answer']
        predicted_confidence = row['probability']

        if index in incorrect_prediction_indexes:
            incorrect_prediction_probabilities.append(predicted_confidence)
            image_path = validation_dict[image_id]

            fig = create_image_figure(image_path=image_path, predicted_breed=predicted_breed,
                                      predicted_confidence=predicted_confidence, actual_breed=actual_breed)
            file_name = actual_breed + '_mistaken_for_' + predicted_breed + '_' + image_id + '.png'
            store_figure(fig, classifier_name, incorrect_image_file_path, file_name)
            plt.close(fig)
        else:
            correct_prediction_probabilities.append(predicted_confidence)
            image_path = validation_dict[image_id]

            fig = create_image_figure(image_path=image_path, predicted_breed=predicted_breed,
                                      predicted_confidence=predicted_confidence)
            file_name = predicted_breed + '_' + image_id + '.png'
            store_figure(fig, classifier_name, correct_image_file_path, file_name)
            plt.close(fig)

    distribution_plot = sns.distplot(correct_prediction_probabilities, kde=False, rug=True, axlabel="ラベル推定確率")
    distribution_plot.set_title("正解ラベルデータ確率分布")
    store_figure(distribution_plot.get_figure(), classifier_name, maps_file_path,
                 VALIDATION_CORRECT_PREDICTION_DISTRIBUTION_PLOT_FILE_NAME)

    plt.show()

    distribution_plot = sns.distplot(incorrect_prediction_probabilities, kde=False, rug=True, axlabel="ラベル推定確率")
    distribution_plot.set_title("不正解ラベルデータ確率分布")
    store_figure(distribution_plot.get_figure(), classifier_name, maps_file_path,
                 VALIDATION_INCORRECT_PREDICTION_DISTRIBUTION_PLOT_FILE_NAME)
    plt.show()

    confidence_level = 0.9

    confident_incorrect_count = len([x for x in incorrect_prediction_probabilities if x > confidence_level])
    confident_correct_count = len([x for x in correct_prediction_probabilities if x > confidence_level])
    confident_incorrect = confident_incorrect_count / len(incorrect_prediction_probabilities)
    confident_correct = confident_correct_count / len(correct_prediction_probabilities)

    print("Amount of incorrect label probabilities with confidence over " + str(
        confidence_level) + " : " + str(confident_incorrect_count) + "(" + str((confident_incorrect * 100)) + "%)")
    print("Amount of correct label probabilities with confidence over " + str(
        confidence_level) + " : " + str(confident_correct_count) + "(" + str((confident_correct * 100)) + "%)")
