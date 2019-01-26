# Pet Breed Classifier

Ever wondered what dog breed you resemble the most?<br>
This is a Python implementation of a pet breed classifier.
It uses transfer learning from Deep Neural Networks that were trained on Imagenet.
A logistic regression is used to map the final layer.
<br>
The model can be set to use xception, inception or bost for generating bottleneck features.
Running this program on all data might be quite slow, consider tweaking the training proportion parameter when running.

## Getting Started

The executable scripts are per below:

* **tools/download_stanford_dog_data.py** - Download Stanford Dog Data<br>
Run this to download the data, extract it and store it for usage.<br>

* **scripts/train_classifier.py** - Train a classifier with set parameters and store it for reuse.<br>
Classifiers are stored in output/classifiers/[classifier_name].<br>
Warning: Depending on your computer, this may take a while.<br>

* **scripts/validate_classifier.py** - Validate the trained classifier. <br>
For this to be executable, the training_proportion of the specified classifier must be strictly less than 1 and low enough that at least one picture remains in the training set.

* **scripts/analyse_classifier.py** - Apply classifier to a file or folder and store results. <br>
Can only be used if the classifier has training proportion set to <1. <br> 
Results are stored in output/classifiers/[classifier_name].<br>
Warning: Depending on your computer, this may take a while.<br>
Run with -h or --help for all arguments and parameters.

* **scripts/plot_analyser_results.py** - Use stored evaluation result and generate plots. <br>
Results are stored in output/classifiers/[classifier_name].<br>
Run with -h or --help for all arguments and parameters.


Note: For tensorflow GPU support, change tensorflow to tensorflow-gpu in requirements.txt

### Prerequisites

Needs Tensorflow and several other libraries,
please see requirements.
It runs on Windows, Linux and MacOS.

## Running the tests

This project currently has no tests...
This is on my TODO list.

## Built With

Development tools

* [Python 3.6](https://www.python.org/downloads/) - Language runtime
* [PyCharm](https://www.jetbrains.com/pycharm/) - IDE by JetBrains

Key Libraries

* [Tensorflow](https://www.tensorflow.org/) - Deep Neural Network library
* [Keras](https://keras.io/) - Library that wraps Tensorflow (and other DNN libraries) and makes it easier to use
* [Scikit-Learn](https://scikit-learn.org/stable/) - Machine Learning library that is just great

Just to mention a few, see requirements.py for all libraries.

## Authors

* **Krister S Jakobsson** - *Implementation*

## License

This project is licensed under the Boost License - see the [license](LICENSE.md) file for details

## Acknowledgments

* **Kaggle Dog Breed Identification Challenge** -  For giving me the original idea, and various contributers who shared ideas for solutions.
[Original Challenge](https://www.kaggle.com/c/dog-breed-identification)
* **Stanford Dogs Dataset** - For providing the data (Aditya Khosla, Nityananda Jayadevaprakash, Bangpeng Yao, and Fei-Fei Li)
[Original Data](http://vision.stanford.edu/aditya86/ImageNetDogs/)
