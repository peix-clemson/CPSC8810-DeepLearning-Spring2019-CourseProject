## VGG CNN
This is the example code for the course project of CPSC 8810 Deep Learning at Clemson University.

## Introduction
In this project, we intend to implement a CNN for a 10-class classification task to identify nonbullying action images and 9 kinds of bullying action images.

This implementation is a VGG19 with batch norm, xavier initializer and ELU activators. Adam is employed as the optimizer. This implementation can easily achieve near 100% accuracy on the raw training dataset. In order to achieve acceptable generalized performance, several online preprocessing methods are applied to augment the training data. Nonbullying images are randomly picked from Stanford 40 Actions for training, while from PIPA for testing. All test nonbullying images are unseen for the model during training.

VGG19 itself is powerful enough to finish this task. Networks with more complex structures, ResNet152, ResNext152, SENet154 and DenseNet201, were tested but no significant improvement will they bring.

Get access to the model and log file via your clemson account at: https://drive.google.com/drive/folders/1TU76Ur5GWlDmKevMZVhaWX9gFRvfCyrt?usp=sharing

## Usage
Tensorflow is required.

Training:

    python main.py --train --datadir <training data directory>

The nonbullying images are assumed stored inside the folder named `nonbullying` in the `datadir` directory.

Testing:
    
    python main.py <image 1> <image 2> ... <image n>



