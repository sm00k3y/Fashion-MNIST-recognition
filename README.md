# Fashion-MNIST Recognition

## Table of contents
* [Introduction](#introduction)
* [Methods](#methods)
* [Results](#results)
* [Usage](#usage)

## Introduction
The goal of this project is Fashion-MNIST image classification.

Fashion-MNIST is a dataset of Zalando's article images. The dataset consists of:
* Training set of 60,000 examples
* Test set of 10,000 examples.

Each example is a 28x28 pixels greyscale image, associated with a label from 10 classes.

Here is and example of first 40 images in the training set:

![Image yet to be uploaded](readme_data/data_pres.PNG)

Our goal is to develop a model which will classify images in a test set with the best accuracy.

My first attempt will be using a KNN - K Nearest Neighbors model with parameters:
* Distance function: Taxicab Metrics (also known as Manhattan Distance)
* Range of *k* parameter: [0..15]
* Weights: uniform - all distances are weighted equally

My second attempy will be making a CNN - Convolutional Neural Network with Tensorflow framework. The structure and parameters of the network are going to be described in *Methods*.

Project is created mainly with:
* Python: 3.7.3
* Tensorflow: 2.2.0
* Matplotlib: 3.2.1

## Methods
### First model - KNN
KNN is usually quite a simple solution, but many times proves to be also quite effective. Let's look at the most important functions of this model.
```python
def taxicab_metric(X, X_train):
    """
    Method returns a taxicab distance, also known as Manhattan distance between two images
    """
    # Normalizing the data
    X = X / 255.
    X_train = X_train / 255.

    distance = np.empty((len(X), len(X_train)))

    for i in range(np.shape(X)[0]):
        distance[i] = np.sum(abs(X_train - X[i]), axis=1)
        if i % 100 == 0:
            print(100.*i/np.shape(X)[0], "%")   # Print progress every 1%

    return distance
```
### Second model - CNN

## Usage
Usage here

## Results
also here

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install foobar
```

```python
import foobar

foobar.pluralize('word') # returns 'words'
foobar.pluralize('goose') # returns 'geese'
foobar.singularize('phenomena') # returns 'phenomenon'
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)