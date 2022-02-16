# PCA-SVD-Autoencoder-Fourier-Wavelet-Transformation-for-denoising
This is the solution of PCA-SVD-Autoencoder-Fourier-Wavelet-Transformation-for-denoising

Project comes from homework in Wuhan University

Environment for Denoise_Autodecoder: Python 3.5+Opencv-Python 4.2.0.34+tensorflow-cpu or gpu with NVIDIA GeForce GTX 750 Ti

First Update:
Fourier.py, in this function, I use Guassian low-pass and butterworth to denoise.
Transform the image of spatial domain into frequency domain. Pass the major information and filter the rest. 
if you have tried other kinds of methods, you can fork my project and sent your PR to me.

New update:
pca.py. PCA is used to denoise. The idea is find the major information of an image to represent the whole image. 

New update:
Denoise_Autodecoder.py. 
