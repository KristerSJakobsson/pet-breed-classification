from os.path import dirname, abspath, join

ROOT_DIR = join(dirname(abspath(__file__)))  # This is the path to the root
SOURCE_DIR = join(ROOT_DIR, "src")

CLASS_PATH = join(ROOT_DIR, "classes")
RESOURCES_PATH = join(ROOT_DIR, "resources")
OUTPUT_PATH = join(ROOT_DIR, "output")
CLASSIFIERS_PATH = join(OUTPUT_PATH, "classifiers")
TEST_PATH = join(OUTPUT_PATH, "tests")
TEST_DATA_PATH = join(OUTPUT_PATH, "dummy_data")

TRAIN_PATH_FOLDER_NAME = "image_data"
TRAIN_PATH = join(RESOURCES_PATH, TRAIN_PATH_FOLDER_NAME)
