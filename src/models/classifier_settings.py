from enum import Enum
from os.path import join

from definitions import OXFORD_IMAGE_DATA_FOLDER_NAME, STANFORD_IMAGE_DATA_FOLDER_NAME, RESOURCES_PATH


class PretrainedClassifier(Enum):
    INCEPTION = 1
    XCEPTION = 2


class ImageSource(Enum):
    OXFORD = OXFORD_IMAGE_DATA_FOLDER_NAME
    STANFORD = STANFORD_IMAGE_DATA_FOLDER_NAME

    def getResourceFolder(self):
        return join(RESOURCES_PATH, self.value)
