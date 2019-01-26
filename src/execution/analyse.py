from os import listdir
from os.path import basename, join

import matplotlib.pyplot as plt

from settings import INPUT_SIZE
from src.models.logreg_classifier import ExistingClassifier
from src.models.containers import ClassifierDetails
from src.utils.image_utils import load_and_preprocess_multiple_images, load_and_preprocess_image, is_image_file, \
    create_image_figure
from src.utils.io_utils import is_file, is_directory, store_dataframe, store_figure


def analyse_image(image_path: str, classifier_details: ClassifierDetails):
    classifier_name = classifier_details.get_name()

    size = (INPUT_SIZE, INPUT_SIZE)
    all_image_paths = []
    if is_file(image_path):
        image_data = load_and_preprocess_image(image_path, size)
        all_image_paths.append((basename(image_path), image_path))
    elif is_directory(image_path):
        all_files = listdir(image_path)
        for file in all_files:
            file_path = join(image_path, file)
            if is_image_file(file_path):
                all_image_paths.append((file, file_path))
        image_data = load_and_preprocess_multiple_images([value[1] for value in all_image_paths], size)

    else:
        raise FileNotFoundError("Could not locate any image files with " + image_path)

    machine_learner = ExistingClassifier(classifier_name=classifier_name)
    classifier_results = machine_learner.apply_to_stored_learner(image_data, [value[0] for value in all_image_paths])

    predictions = classifier_results.get_prediction()
    store_dataframe(predictions, classifier_name, 'analyse_predictions.csv')

    probability = classifier_results.get_probability()
    store_dataframe(probability, classifier_name, 'analyse_probabilities.csv')

    image_file_path = "analyse_labeled"
    for index, row in predictions.iterrows():
        image_id = index
        image_path = [item[1] for item in all_image_paths if item[0] == image_id][0]
        predicted_breed = row['prediction']
        predicted_confidence = float(row['probability'])
        fig = create_image_figure(image_path=image_path,
                                  predicted_breed=predicted_breed,
                                  predicted_confidence=predicted_confidence)
        file_name = image_id + "_" + predicted_breed + '.png'
        store_figure(fig, classifier_name, image_file_path, file_name)
        plt.close(fig)
