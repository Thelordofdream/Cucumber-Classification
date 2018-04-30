# Cucumber-Classification
A cucumber classification

## Data Sets
Here, we use a cucumber image data set from Japan, which contains 9 classes of cucumber.

* [9 Classes Cucumber Image](https://github.com/workpiles/CUCUMBER-9)

The details of CUCUMBER-9 is the following:

* Each cucumber contains three images, including above, below and side
* The size of cucumber image is 32 x 32
* Each image have three channels (RGB)
* There are 10 classes, inculding 0 class ("others")

## Main Task
* Load the cucumber dataset
* Build classification model
* Train the classifier and tune hyperparameter
* Evaluate the performance of the model
* Implement an end-to-end cucumber classifier
* Realise data visualization

## Model

* ResNet 21
* Linear Classifier

## Language & Framework

* Python 3.6
	* Object oriented	
* TensorFlow 1.7
	* TensorBoard
   * TFRecord