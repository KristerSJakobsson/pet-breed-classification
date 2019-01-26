# Original resources from University of Oxford:
# Data link: http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
# OBS: Data is approx 800 MB

import tarfile
import re

from os.path import exists, join
from os import remove, listdir, makedirs
from shutil import move, rmtree
from urllib.request import urlretrieve

from definitions import RESOURCES_PATH, TRAIN_PATH_FOLDER_NAME
from src.utils.image_utils import is_image_file

DOG_CAT_DATA_URL = 'http://www.robots.ox.ac.uk/~vgg/data/pets/'
DOG_CAT_DATA_FILENAME = "images.tar.gz"

oxford_data_targzfile = join(RESOURCES_PATH, DOG_CAT_DATA_FILENAME)
targzfile_internal_folder_name = "images"

oxford_extracted_data_default_folder = join(RESOURCES_PATH, targzfile_internal_folder_name)
oxford_extracted_data_final_folder = join(RESOURCES_PATH, TRAIN_PATH_FOLDER_NAME)

# See if we already have data downloaded, if so, tell user to manually delete
if not exists(oxford_extracted_data_default_folder) and not exists(oxford_extracted_data_final_folder):
    filename, headers = urlretrieve(DOG_CAT_DATA_URL, oxford_data_targzfile)
else:
    print(
        "Folders or files with names " + TRAIN_PATH_FOLDER_NAME + "/" + targzfile_internal_folder_name + " already exists, please delete them to run this script.")

# Extract data into folder
file = tarfile.open(name=oxford_data_targzfile)
file.extractall(path=RESOURCES_PATH)
remove(path=oxford_data_targzfile)

files = listdir(oxford_extracted_data_default_folder)

filename_regex = re.compile(
    r'^(n(?P<breed>\w+)_(?P<id>\d+))$',
    re.IGNORECASE)

breed_path_dictionary = {}

# First, we store all pictures in a hashmap so that: breed -> list of images
for index, value in files:
    if is_image_file(value) and filename_regex.match(value):
        matches = filename_regex.search(value)
        path_id = matches.group('id')
        path_breed = matches.group('breed')
        if path_breed not in breed_path_dictionary.keys():
            breed_path_dictionary[path_breed] = []
        breed_path_dictionary[path_breed].append(value)

# Crate a folder for each breed with format [breed_id]-[breed_name] and move all files there
breed_incremental_index = 1
for key in breed_path_dictionary.keys():
    breed = key
    id = breed_incremental_index

    breed_path = join(oxford_extracted_data_final_folder, str(id) + "-" + breed)
    makedirs(path=breed_path)

    # Move all images from /images to /image_data/[breed_path]/[image_name]
    for index, image in breed_path_dictionary[key]:
        image_old_path = join(oxford_extracted_data_default_folder, image)
        image_new_path = join(breed_path, image)
        move(src=image_old_path, dst=image_new_path)

    breed_incremental_index = breed_incremental_index + 1

# Delete all files that are remaining in /images
rmtree(path=oxford_extracted_data_default_folder)