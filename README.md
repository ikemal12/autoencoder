# Autoencoder

## Overview 

The aim of this project was to develop an autoencoder (a type of unsupervised neural network) to perform lossy image compression. The model was trained on a dataset of ~1200 images with dimensions 150 x 225 (x3 for RGB channels).

## Structure 

1. Encoder: Compresses the input image into a lower-dimensional latent representation.
2. Latent space (Bottleneck): This is the compressed representation of the input.
3. Decoder: Reconstructs the original input from the latent space, with some information lost.

## Architecture 

Initially I experimented with a fully connected model using linear layers, but found it struggled to capture spatial hierarchies within the image data which led to poor reconstruction quality. Hence I adopted a convolutional-based (CNN) architecture which preserves local spatial structures and extracts hierarchical features. Each convolutional layer is followed by batch normalisation to stabilise training and a LeakyReLU activation function. Instead of max pooling, I opted for stride-based downsampling to retain spatial information, as well as padding to prevent misalignment.

The decoder mirrors the encoder, using transpose convolutions for upsampling, though I refined kernel sizes and strides to mitigate checkerboard artifacts. Additionally, residual connections and self-attention in the bottleneck layer helped retain fine-grained details and capture long-range dependencies. 

## Hyperparameter Tuning



## Training 

My model utilises the Lion optimiser, a momentum-based method with fewer parameters. This resulted in smoother convergence and better generalisation. It also reduced fluctuations in validation loss in later epochs.

I also implemented learning rate scheduling, initially using step decay but later switching to cosine annealing. This was helpful in preventing large oscillations in weight updates and therefore stabilising the training process. Without scheduling, I observed slower convergence and suboptimal reconstructions.

An issue that arose in the deeper layers of my model was gradient explosion - to combat this I incorporated gradient clipping, which ensured stable optimisation, especially at larger batch sizes.

## Results



## Applications 