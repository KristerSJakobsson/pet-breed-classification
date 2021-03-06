# Pet Breed Classifier

Ever wondered what pet breed is in a picture?<br>
Then this tool might be of help to you. <br>
What breeds can be detected depend on training data-set, <br> 
currently the Stanford data contains 120 breeds of dogs and the Oxford data set 25 breeds of dogs and 12 breds of cats. <br>
<br>
This is a Python implementation of a pet breed classifier.
It uses transfer learning from Deep Neural Networks that were trained on Imagenet.
A logistic regression is used to map the final layer.
<br>
The model can be set to use xception, inception or both for generating bottleneck features.
Running this program on all data might be quite slow, consider tweaking the training proportion parameter when running.

## Getting Started

I recommend setting up a project in PyCharm and pulling the sources from git.
Run 'pip install requirements.txt' to install all required packages.
Go to Edit Configurations in PyCharm and specify the below scripts.
(PyCharm will add the project root to PYTHONPATH, otherwise you will have to do this manually.)

The executable scripts are per below:

* **tools/download_stanford_dog_data.py** - Download Stanford Dog Data (~ 750 MB)<br>
Run this to download the data, extract it and store it for usage.<br>

* **tools/download_oxford_cat_dog_data.py** - Download Oxford-IIIT Pet Dataset (~ 800 MB)<br>
Run this to download the data, extract it and store it for usage.<br>

* **scripts/train_classifier.py** - Train a classifier with set parameters and store it for reuse.<br>
Classifiers are stored in output/classifiers/[classifier_name].<br>
Warning: Depending on your computer, this may take a while.<br>

* **scripts/use_classifier.py** - Apply classifier to a file or folder and store results. <br>
Can only be used if the classifier has training proportion set to <1. <br> 
Results are stored in output/classifiers/[classifier_name].<br>
Warning: Depending on your computer, this may take a while.<br>
Run with -h or --help for all arguments and parameters.

* **scripts/validate_classifier.py** - Validate the trained classifier. <br>
For this to be executable, the training_proportion of the specified classifier must be strictly less than 1 and low enough that at least one picture remains in the training set.

* **scripts/validate_plot_results.py** - Use stored evaluation result and generate plots (distribution plot, clustermap and heatmap). <br>
Results are stored in output/classifiers/[classifier_name].<br>
Run with -h or --help for all arguments and parameters.

Note 1: The above scripts require parameters on execution. Please supply the same parameters when you train and use the same classifier.
Note 2: For Tensorflow GPU support, change tensorflow to tensorflow-gpu in requirements.txt. Without it, classification can be quite slow.

### Prerequisites

Needs Tensorflow and several other libraries,
please run pip install requirements.txt
It runs on Windows, Linux and MacOS.
Note that training on the full data will take a long time without GPU.

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

* **Krister S Jakobsson** - *Implementation and pretty much everything else*

## License

This project is licensed under the Boost License - see the [license](LICENSE.md) file for details

## Acknowledgments

* **Kaggle Dog Breed Identification Challenge** -  For giving me the original idea, and various contributers who shared ideas for solutions.
[Original Challenge](https://www.kaggle.com/c/dog-breed-identification)
* **Stanford Dogs Dataset** - For providing dog breed data (Aditya Khosla, Nityananda Jayadevaprakash, Bangpeng Yao, and Fei-Fei Li) [Original Data](http://vision.stanford.edu/aditya86/ImageNetDogs/)
* **Oxford-IIIT Pet Dataset** - For providing dog breed and cat breed data (Omkar M Parkhi and Andrea Vedaldi, and Andrew Zisserman and C. V. Jawahar) [Original data](http://www.robots.ox.ac.uk/~vgg/data/pets/)