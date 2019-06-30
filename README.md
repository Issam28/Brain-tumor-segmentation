# Fully automatic Brain tumor segmentation

### Brief overview

This repository provides source code for a deep convolutional neural network architecture designed for brain tumor segmentation with BraTS2017 dataset. 
The architecture is fully convolutional network (FCN) built upon the well-known U-net model and it makes use of residual units instead of plain units to speedup training and convergence.
The Brain tumor segmentation problem exhibits severe class imbalance where the healthy voxels comprise 98% of total voxels,0.18% belongs to necrosis ,1.1% to edema and non-enhanced and 0.38% to enhanced tumor. 
The issue is addressed by: 1) adopting a patch-based training approach; 2) using a custom loss function that accounts for the imbalance. 
During training, 2D patches of size 128x128 from the axial plane are randomly sampled. And by doing so it allows to dismiss patches from pixels with zero intensity and therefore it helps a bit to alleviate the problem.

The implementation is based on keras and tested on both Theano and Tensorflow backends.

Here are some results predicted by a model trained for 2 epochs :

*   **HGG cases** :

![Optional Text](../master/docs/images/HGG-Brats17_2013_7_1-111.png)
![Optional Text](../master/docs/images/HGG-Brats17_CBICA_ASV_1-88.png)
![Optional Text](../master/docs/images/HGG-Brats17_TCIA_186_1-90.png)

*   **LGG cases** :

![Optional Text](../master/docs/images/LGG-Brats17_TCIA_202_1-70.png)
![Optional Text](../master/docs/images/LGG-Brats17_2013_24_1-91.png)
![Optional Text](../master/docs/images/LGG-Brats17_TCIA_462_1-97.png)

### Requirements

To run the code, you first need to install the following prerequisites: 

* Python 3.5 or above
* numpy
* keras
* scipy
* SimpleITK

### How to run

1. Execute first `extract_patches.py` to prepare the training and validation datasets.
2. then `train.py` to train the model.
3. `predict.py` to make final predictions.

```
python extract_patches.py
python train.py
python predict.py
```
### How to cite 

This code is an implementation of [this paper](https://link.springer.com/chapter/10.1007/978-3-030-11726-9_4). If you find this code useful in your research, please consider citing: 

```
@inproceedings{kermi2018deep,
  title={Deep Convolutional Neural Networks Using U-Net for Automatic Brain Tumor Segmentation in Multimodal MRI Volumes},
  author={Kermi, Adel and Mahmoudi, Issam and Khadir, Mohamed Tarek},
  booktitle={International MICCAI Brainlesion Workshop},
  pages={37--48},
  year={2018},
  organization={Springer}
}
```
