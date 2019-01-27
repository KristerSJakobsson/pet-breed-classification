from os import makedirs
from os.path import expanduser, exists, join
from typing import List, Any

from numpy import ndarray, hstack
from keras.applications import inception_v3
from keras.applications import xception
from sklearn.linear_model import LogisticRegression

from settings import TRAINING_CLASSIFIER_FILE_NAME, TRAINING_CLASSIFIER_DATA_FILE_NAME, DEFAULT_POOLING
from .classifier_settings import PretrainedClassifier, ImageSource
from .containers import ClassifierDetails, ClassifierResult
from src.utils.io_utils import store_sklearn_classifier, load_sklearn_classifier, store_serializable_object, \
    load_serializable_object


class ClassifierBase:
    def __init__(self, classifier_details: ClassifierDetails):
        """
        Base constructor. Given details for a classifier, store relevant settings.
        Also, prepare cache dirs for Keras
        (first time executed, keras will download and cache the pretrained neural networks used)
        :param classifier_details: Parameters for the classifier
        """
        keras_cache_dir = expanduser(join('~', '.keras'))
        if not exists(keras_cache_dir):
            makedirs(keras_cache_dir)
        keras_models_dir = join(keras_cache_dir, 'models')
        if not exists(keras_models_dir):
            makedirs(keras_models_dir)

        self._pooling = DEFAULT_POOLING
        self._classifier_objects = None

        self._classifier_details = classifier_details
        self.update_classifier_objects()


    @property
    def pooling(self):
        return self._pooling

    @pooling.setter
    def pooling(self, value: str):
        self._pooling = value
        self.update_classifier_objects()

    @property
    def classifier_details(self):
        return self._classifier_details

    def update_classifier_objects(self):
        """
        Makes sure classifier objects are updated accordingly
        """
        classifier_objects = []
        for classifier in self._classifier_details.training_classifiers:
            if classifier == PretrainedClassifier.XCEPTION:
                classifier_objects.append(
                    xception.Xception(weights='imagenet', include_top=False, pooling=self._pooling))
            elif classifier == PretrainedClassifier.INCEPTION:
                classifier_objects.append(
                    inception_v3.InceptionV3(weights='imagenet', include_top=False, pooling=self._pooling))
            else:
                raise ValueError("Unsupported classifier object " + classifier.name.lower())
        self._classifier_objects = classifier_objects

    @staticmethod
    def _generate_bottleneck_features_for_classifiers(data: ndarray, classifiers: List[Any]) -> ndarray:
        """
        Creates the bottleneck features for the specified data with multiple classifiers.
        :param data: The matrix representation of the image data we need bottleneck features for
        :param classifiers: List of classifiers we want to generate features for
        :return: A matrix with bottleneck features
        """
        results = []
        for classifier in classifiers:
            results.append(classifier.predict(data, batch_size=32, verbose=1))
        return hstack(results)


class NewClassifier(ClassifierBase):
    def __init__(self, breed_names: List[str], training_classifiers: List[PretrainedClassifier], seed: int,
                 training_proportion: float, image_sources: List[ImageSource]):
        """
        Creates a classifier that does not already exist. (it needs to be trained)
        :param breed_names: List of breed names
        :param training_classifiers: Represents a list of pretrained classifiers used internally
        :param seed: The seed value for the classifier
        :param training_proportion: The proportion of data used to train
        :param image_sources: Image sources for classifier
        """
        classifier_details = ClassifierDetails(breed_names, training_classifiers, seed, training_proportion, image_sources)
        super().__init__(classifier_details)

    def train_and_store_classifier(self, x_training: ndarray, y_training: ndarray,
                                   classifier_details: ClassifierDetails):
        """
        Train the classifier and store for reuse
        :param x_training: The image data
        :param y_training: The image classifications
        :param classifier_details: Object representing the classifier we want to use
        :param seed: The seed value for the logistic regression
        """

        trained_bottleneck_features = self._generate_bottleneck_features_for_classifiers(
            data=x_training, classifiers=self._classifier_objects)

        logistic_regression_classifier = LogisticRegression(multi_class='multinomial',
                                                            solver='lbfgs',
                                                            random_state=classifier_details.seed)
        number_of_classes = y_training.shape[1]
        column_index_encoded_dataframe = y_training * range(number_of_classes)
        logistic_regression_classifier.fit(X=trained_bottleneck_features,
                                           y=column_index_encoded_dataframe.sum(axis=1))

        classifier_name = self._classifier_details.get_name()

        # Store logistic classifier
        store_sklearn_classifier(sklearn_classifier=logistic_regression_classifier,
                                 classifier_name=classifier_name,
                                 filename=TRAINING_CLASSIFIER_FILE_NAME)

        # Store classifier details
        store_serializable_object(serializable_object=classifier_details,
                                  classifier_name=classifier_name,
                                  filename=TRAINING_CLASSIFIER_DATA_FILE_NAME)


class ExistingClassifier(ClassifierBase):

    def __init__(self, classifier_name: str):
        """
        Loads a classifier that does already exist. (it has already been trained)
        classifier_name: The name of the classifier to load
        """
        classifier_details = self.get_classifier_details(classifier_name)
        super().__init__(classifier_details)

    def get_classifier_details(self, classifier_name: str) -> ClassifierDetails:
        return load_serializable_object(classifier_name=classifier_name,
                                        filename=TRAINING_CLASSIFIER_DATA_FILE_NAME)

    def get_classifier_object(self) -> Any:
        return load_sklearn_classifier(classifier_name=self._classifier_details.get_name(),
                                       filename=TRAINING_CLASSIFIER_FILE_NAME)

    def apply_to_stored_classifier(self, image_data: Any, images: List[str]) -> ClassifierResult:
        """
        Loads stored classifier and applies it to image data
        :param image_data: The matrix representation of the data
        :param images: A list of the names for the images in image_data
        :return: The result of the classification
        """
        # Retrieve the stored logreg
        logreg = self.get_classifier_object()

        # Generate features for the new resources
        data_bottlenect_features = self._generate_bottleneck_features_for_classifiers(image_data,
                                                                                      self._classifier_objects)

        # Apply logreg to features and predict
        prediction = logreg.predict(data_bottlenect_features)
        probabilities = logreg.predict_proba(data_bottlenect_features)

        return ClassifierResult(probabilities, prediction, images, self._classifier_details)
