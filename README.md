# Liver-segmentation
CNN for liver segmentation with DCE-MR images.

The six phases of the dynamic contrast enhanced (DCE) MRI are used as input images. The six phases are concatenated to one image with six channels. 

Code is based on Python 3.6, tensorflow, and keras.

The script is not adjusted for general use. Directories and data should be supplied by the user.

Based on:
Mariëlle J. A. Jansen, Hugo J. Kuijf, and Josien P. W. Pluim "Optimal input configuration of dynamic contrast enhanced MRI in convolutional neural networks for liver segmentation", Proc. SPIE 10949, Medical Imaging 2019: Image Processing, 109491V (15 March 2019); https://doi.org/10.1117/12.2506770 
