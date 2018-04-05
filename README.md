# Fully automatic Brain tumor segmentation

### Brief overview

This repository provides source code for a deep convolutional neural network architecture designed for brain tumor segmentation with BraTS2017 dataset. 
The architecture is fully convolutional network (FCN) built upon the well-known U-net model and it makes use of residual units instead of plain units to speedup training and convergence.
The Brain tumor segmentation problem exhibits severe class imbalance where the healthy voxels comprise 98% of total voxels,0.18% belongs to necrosis ,1.1% to edema and non-enhanced and 0.38% to enhanced tumor. 
The issue is addressed by: 1) adopting a patch-based training approach; 2) using a custom loss function that accounts for the imbalance. 
During training, 2D patches of size 128x128 from the axial plane are randomly sampled. And by doing so it allows to dismiss patches from pixels with zero intensity and therefore it helps a bit to alleviate the problem.

The implementation is based on keras and tested on both Theano and Tensorflow backends.

### Requirements

To run the code, you first need to install the following prerequisites: 

* Python 3.5 or above
* numpy
* keras
* scipy
* SimpleITK
* scikit-image

### How to run

* [1] Execute first extract_patches.py to prepare the training and validation datasets.
* [2] then train.py to train the model.
* [3] predict.py to make final predictions.
