from os import listdir
from os.path import basename, join

import matplotlib.pyplot as plt

from settings import INPUT_SIZE, ANALYSE_PREDICTIONS_FILE_NAME, ANALYSE_PROBABILITIES_FILE_NAME
from src.models.logreg_classifier import ExistingClassifier
from src.models.containers import ClassifierDetails
from src.utils.image_utils import load_and_preprocess_multiple_images, load_and_preprocess_image, \
    create_image_figure
from src.utils.io_utils import is_directory, is_image_file, store_dataframe, store_figure


def analyse_image(image_path: str, classifier_details: ClassifierDetails):
    classifier_name = classifier_details.get_name()

    size = (INPUT_SIZE, INPUT_SIZE)
    all_image_paths = []
    if is_directory(image_path):
        all_files = listdir(image_path)
        for file in all_files:
            file_path = join(image_path, file)
            if is_image_file(file_path):
                all_image_paths.append((file, file_path))
        image_data = load_and_preprocess_multiple_images([value[1] for value in all_image_paths], size)
    elif is_image_file(image_path):
        image_data = load_and_preprocess_image(image_path, size)
        all_image_paths.append((basename(image_path), image_path))
    else:
        raise FileNotFoundError("Could not locate any image at " + image_path)

    machine_learner = ExistingClassifier(classifier_name=classifier_name)
    image_basenames = [value[0] for value in all_image_paths]
    classifier_results = machine_learner.apply_to_stored_classifier(image_data=image_data,
                                                                    images=image_basenames)

    predictions = classifier_results.get_prediction()
    store_dataframe(predictions, classifier_name, ANALYSE_PREDICTIONS_FILE_NAME)

    probability = classifier_results.get_probability()
    store_dataframe(probability, classifier_name, ANALYSE_PROBABILITIES_FILE_NAME)

    image_file_path = "analyse_labeled"
    for index, row in predictions.iterrows():
        image_id = index
        image_path = [value[1] for value in all_image_paths if value[0] == image_id][0]
        predicted_breed = row['prediction']
        predicted_confidence = float(row['probability'])
        fig = create_image_figure(image_path=image_path,
                                  predicted_breed=predicted_breed,
                                  predicted_confidence=predicted_confidence)
        file_name = image_id + "_" + predicted_breed + '.png'
        store_figure(fig, classifier_name, image_file_path, file_name)
        plt.close(fig)
