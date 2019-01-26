from pandas import value_counts

from settings import INPUT_SIZE
from src.models.data_wrangler import DataWrangler
from src.models.logreg_classifier import NewClassifier
from src.models.containers import ClassifierDetails
from src.utils.image_utils import create_breed_bar_graph
from src.utils.io_utils import store_serializable_object, store_figure
from src.utils.training_utils import load_data_and_split_for_training_validating


def train_classifier(classifier_details: ClassifierDetails):
    classifiers = classifier_details.training_classifiers
    seed = classifier_details.seed
    training_proportion = classifier_details.training_proportion
    breeds_list = classifier_details.training_breed_names

    # Read stanford train resources
    training_data, validation_data = load_data_and_split_for_training_validating(breeds_list, training_proportion, seed)

    # Pre-process resources (randomize order, load images, store in numpy arrays)
    data_wrangler = DataWrangler(image_files=training_data)
    data_wrangler.seed = seed
    data_wrangler.image_input_size = INPUT_SIZE
    data_wrangler.execute_load_training_data(shuffle=True,
                                             breeds_list=breeds_list)

    x_training_data = data_wrangler.x_data
    y_training_data = data_wrangler.y_data

    # Plot how many images of each breed we have
    image_classifications = value_counts(
        [data.image_classification for data in training_data],
        ascending=True)

    breeds_bar_figure = create_breed_bar_graph(image_classifications)

    # Create classifier
    machine_learner = NewClassifier(breed_names=breeds_list,
                                    training_classifiers=classifiers,
                                    seed=seed,
                                    training_proportion=training_proportion)

    # Train classifier
    machine_learner.train_and_store_classifier(
        x_training=x_training_data,
        y_training=y_training_data,
        classifier_details=classifier_details)

    classifier_name = machine_learner.classifier_details.get_name()

    # Store figure
    store_figure(figure=breeds_bar_figure,
                 classifier_name=classifier_name,
                 image_folder="data",
                 filename="breed_data_proportions.png")

    # Store validation resources in classifier
    store_serializable_object(serializable_object=validation_data,
                              filename="validation_data.json",
                              classifier_name=classifier_name)
