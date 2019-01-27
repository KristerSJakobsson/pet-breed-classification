from typing import Tuple, Any, List

import matplotlib.pyplot as plt
from numpy import zeros, expand_dims
from pandas import DataFrame
from keras.applications import xception
from keras.preprocessing import image
from numpy import ndarray
from tqdm import tqdm


def load_and_preprocess_image(path: str, size: Tuple[int, int]) -> ndarray:
    """
    Loads a specific image and transforms it into a matrix representation (x, y, RGB)
    :param path: Path where the image is
    :param size: The dimensions the image will be scaled to
    :return: Matrix representation of scaled image
    """
    loaded_image = read_single_image(path, size)
    return preprocess_image(loaded_image)


def load_and_preprocess_multiple_images(image_paths: List[str], image_size: Tuple[int, int]) -> ndarray:
    """
    Loads all images from paths and transforms them into matrix representations (index, x, y, RGB)
    :param image_paths: A list of image paths
    :param image_size: The dimensions the image will be scaled to
    :return: Matrix representation of all images
    """
    number_of_images = len(image_paths)
    height = image_size[0]
    width = image_size[1]
    channels = 3

    x_result = zeros(
        shape=(number_of_images, width, height, channels),
        dtype='float32')
    # TODO: The below seems suboptimal... Ok for now
    for index, image_path in tqdm(enumerate(image_paths)):
        x_result[index] = load_and_preprocess_image(image_path, image_size)
    return x_result


def read_single_image(image_path: str, image_size: Tuple[int, int]) -> ndarray:
    """
    Reads a single image from path
    :param image_path: A path to an image
    :param image_size: The dimensions the image will be scaled to
    :return: Matrix representation of scaled image
    """
    loaded_image = image.load_img(path=image_path,
                                  target_size=image_size)
    if loaded_image is None:
        raise FileNotFoundError("Could not load image file: " + image_path)
    return image.img_to_array(loaded_image)


def preprocess_image(image_data: ndarray) -> Any:
    """
    Preprocesses a numpy array encoding a batch of images with Keras library.
    :param image_data: The numpy array of images
    :return: Returns preprocessed images for use with Keras
    """
    return xception.preprocess_input(expand_dims(image_data.copy(), axis=0))

def create_image_figure(image_path: str,
                        predicted_breed: str,
                        predicted_confidence: float,
                        actual_breed: str = None) -> Any:
    """
    Creates and returns a figure object with labels representing prediction result
    :param image_path: The path to the image
    :param predicted_breed: The predicted breed
    :param predicted_confidence: Confidence level (0 to 1)
    :param actual_breed: If known, shows the actual breed (for comparison)
    :return: A figure object corresponding to the input
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    img = read_single_image(image_path, (500, 500))
    ax.imshow(img / 255.)
    ax.text(10, 490, 'Prediction: %s (%s%%)' % (predicted_breed, str(round(predicted_confidence * 100, 2))),
            style='italic',
            bbox={'boxstyle': 'round',
                  'edgecolor': '#690017',
                  'facecolor': '#ffffff'})

    if actual_breed is not None:
        ax.text(10, 456, 'Actual: %s' % actual_breed,
                style='italic',
                bbox={'boxstyle': 'round',
                      'edgecolor': '#005e13',
                      'facecolor': '#ffffff'})
    ax.axis('off')
    return fig


def create_breed_bar_graph(breeds: DataFrame) -> Any:
    """
    Shows a breed bar graph with
    :param breeds:
    :return:
    """
    graph_size = (50, 100)
    ax = breeds.plot(kind='barh', fontsize='40', title='Breed', figsize=graph_size)
    fig = ax.get_figure()

    ax.set(xlabel="Pictures per breed",
           ylabel="Breed")
    ax.xaxis.label.set_size(40)
    ax.yaxis.label.set_size(40)
    ax.title.set_size(60)

    return fig