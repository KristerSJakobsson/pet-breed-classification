# Original resources from Stanford University:
# Annotations link: http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar
# Data link: http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar
# OBS: Data is approx 750 MB

import tarfile

from os.path import exists, join
from os import remove
from shutil import move

from urllib.request import urlretrieve

from definitions import RESOURCES_PATH, STANFORD_IMAGE_DATA_FOLDER_NAME

DOG_DATA_URL = 'http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar'
DOG_DATA_FILENAME = "images.tar"

stanford_data_tarfile = join(RESOURCES_PATH, DOG_DATA_FILENAME)
tarfile_internal_folder_name = "Images"

stanford_extracted_data_default_folder = join(RESOURCES_PATH, tarfile_internal_folder_name)
stanford_extracted_data_final_folder = join(RESOURCES_PATH, STANFORD_IMAGE_DATA_FOLDER_NAME)

# See if we already have data downloaded, if so, tell user to manually delete
if not exists(stanford_extracted_data_default_folder) and not exists(stanford_extracted_data_final_folder):
    filename, headers = urlretrieve(DOG_DATA_URL, stanford_data_tarfile)
else:
    print(
        "Folders or files with names " + STANFORD_IMAGE_DATA_FOLDER_NAME + "/" + tarfile_internal_folder_name + "already exists, please delete them to run this script.")

# Extract data into folder
file = tarfile.open(name=stanford_data_tarfile)
file.extractall(path=RESOURCES_PATH)
move(src=stanford_extracted_data_default_folder, dst=stanford_extracted_data_final_folder)
remove(path=stanford_data_tarfile)
