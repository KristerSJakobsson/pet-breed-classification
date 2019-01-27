# Original resources from University of Oxford:
# Data link: http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
# OBS: Data is approx 800 MB

import tarfile
import re

from os.path import exists, join
from os import remove, listdir, makedirs
from shutil import move, rmtree
from urllib.request import urlretrieve

from definitions import RESOURCES_PATH, OXFORD_IMAGE_DATA_FOLDER_NAME
from src.utils.io_utils import is_image_file

DOG_CAT_DATA_URL = 'http://www.robots.ox.ac.uk/~vgg/data/pets/'
DOG_CAT_DATA_FILENAME = "images.tar.gz"

oxford_data_targzfile = join(RESOURCES_PATH, DOG_CAT_DATA_FILENAME)
targzfile_internal_folder_name = "images"

oxford_extracted_data_default_folder = join(RESOURCES_PATH, targzfile_internal_folder_name)
oxford_extracted_data_final_folder = join(RESOURCES_PATH, OXFORD_IMAGE_DATA_FOLDER_NAME)

# See if we already have data downloaded, if so, tell user to manually delete
if not exists(oxford_extracted_data_default_folder) and not exists(oxford_extracted_data_final_folder):
    filename, headers = urlretrieve(DOG_CAT_DATA_URL, oxford_data_targzfile)
else:
   print(
       "Folders or files with names " + OXFORD_IMAGE_DATA_FOLDER_NAME + "/" + targzfile_internal_folder_name +
       " already exists, please delete them to run this script.")

# Extract data into folder
file = tarfile.open(name=oxford_data_targzfile)
file.extractall(path=RESOURCES_PATH)

files = listdir(oxford_extracted_data_default_folder)

filename_regex = re.compile(
    r'^(?P<breed>\w+)_(?P<id>\d+)\..*$',
    re.IGNORECASE)

breed_path_dictionary = {}

# First, we store all pictures in a hashmap so that: breed -> list of images
for image_file_name in files:
    image_file_path = join(oxford_extracted_data_default_folder, image_file_name)
    if is_image_file(image_file_path) and filename_regex.match(image_file_name):
        matches = filename_regex.search(image_file_name)
        path_id = matches.group('id')
        path_breed = matches.group('breed')
        if path_breed not in breed_path_dictionary.keys():
            breed_path_dictionary[path_breed] = []
        breed_path_dictionary[path_breed].append((image_file_name, image_file_path))

# Crate a folder for each breed with format [breed_id]-[breed_name] and move all files there
breed_incremental_index = 1
for breed_type in breed_path_dictionary.keys():
    breed = breed_type
    id = breed_incremental_index

    breed_folder = str(id) + "-" + breed
    breed_path = join(oxford_extracted_data_final_folder, breed_folder)
    makedirs(breed_path)

    # Move all images from /images to /image_data/[breed_path]/[image_name]
    for image in breed_path_dictionary[breed_type]:
        image_old_path = image[1]
        image_new_path = join(breed_path, image[0])
        move(src=image_old_path, dst=image_new_path)

    breed_incremental_index = breed_incremental_index + 1

# Delete tar file and all files that are remaining in /images
remove(path=oxford_data_targzfile)
rmtree(path=oxford_extracted_data_default_folder)
