# Capstone Project Machine Learning Engineer Nanodegree
This repository is for capstone project for Machine Learning Engineer Nanodegree.

In this project, I trained a model that can classify images with **Convolutional Neural Network**(CNN).
When building a model, hyper-parameter tuning is one of the most essential and difficult tasks.
In this project, I optimized hyper-parameters by utilizing Bayesian Optimization.

## Dataset
The data I used in this project is available [here](https://www.cs.toronto.edu/~kriz/cifar.html)

The short description of the dataset is below.

"The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. 

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class."



## Dependencies
* Numpy
* Scilit-learn
* Seaborn
* Theano
* [Keras](https://github.com/RyosukeHonda/keras)
* [BayesianOptimization](https://github.com/fmfn/BayesianOptimization)
