# SRCNN Paper Summary
Original Paper at [Image Super-Resolution Using Deep Convolutional Networks"](https://arxiv.org/pdf/1501.00092.pdf)
## Introduction
* Prior to this, xisting methods apply Dictionary / Sparse Coding based approach
* Different from the rest - 
    1. Does not explicitly learn the dictionaries or manifolds for modeling the patch space. These  are  implicitly  achieved  via  hidden  layers.
    2. Furthermore, the patch extraction and aggregation are also formulated as convolutional layers, so are involved in the optimization. 
* The entire SR pipeline is fully obtained through learning
* Light Network; Works on CPU
* Fully Feed Forward, no optimization is needed to be solved

## Method
Prior to passing through Convolutional Networks, the image is enlarged to the required factor by Interpolation.
This image (<b>Y</b>) is the input (referred still as "Low Resolution") and the "Ground Truth" SR image <b>X</b> is the Output such that a mapping F(<b>Y</b>) = <b>X</b>

The method consists of 3 sub networks:

### Patch extraction and representation
1. A convolutional layer (W1, B1); n<sub>1</sub> filters each of kernel size f<sub>1</sub> having dimensions, c x f<sub>1</sub>x f<sub>1</sub>; 
2. Activation with Rectified Linear Unit <b>(ReLU,max(0,x))</b>

### Non Linear Mapping
* Mapping the feature of the map W1 is mapped to n<sub>2</sub> dimensional vectors. Equivalent to n<sub>2</sub> filters of size 1x1 ??
    1. W2 contains n2 filters of size n1×f2×f2, and B2 is n2-dimensional. Each of the output n2-dimensional vectors is conceptually a representation of a high-resolution patch that will be used for reconstruction.
    2. Activation again by Rectified Linear Unit <b>(ReLU,max(0,x))</b> 
    
### Reconstruction
* F(Y) = W3 ∗ F2(Y) + B3
* Inspired from the fact that overlapping patches to result in averaging, final filter combines layers of n2 dimesnion to reqired dimenion (c); c filters  of  a  size n2×f3×f3

## Similarity to Sparse Coding ??

## Training
* From Sparse Coding Approach, n1=64, n2=32, f1=9, f2=1, f3=5
* Loss = Mean Square Error (MSE)
* Optimizer = SGD ()
#### Initializing
* Weights are initialized by Gaussian distribution with zero mean and standard deviation 0.001
* Learning rate is 10<sup>-4</sup> for the first two layers, and 10<sup>−5</sup> for the last layer.
<i>To avoid border effects during training, all the convolutional layers have no padding, and the network produces a smaller output.</i>

### Experiments
Upscaling done on patch size 3; 8×10<sup>8</sup> backpropagations
#### Data
1. Training:
    * T91 with patch size = 33; around 24,800 images extracted with a stride = 14
    * ImageNet Dataset, 5 million subimages with patch size 33
2. Validation on Set5/Set14

## Problems
* Sensitive to number of filters in each layer
* Can improve on PSNR but not SSIM/MSSIM values
