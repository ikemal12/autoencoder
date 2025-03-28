# Autoencoder

## Overview 

The aim of this project was to develop an autoencoder (a type of unsupervised neural network) to perform lossy image compression. The model was trained on a dataset of ~1200 images with dimensions 150 x 225 (x3 for RGB channels).

## Structure 

1. Encoder: Compresses the input image into a lower-dimensional latent representation.
2. Latent space (Bottleneck): This is the compressed representation of the input.
3. Decoder: Reconstructs the original input from the latent space, with some information lost.

## Architecture 

Initially I experimented with a fully connected model using linear layers, but found it struggled to capture spatial hierarchies within the image data which led to poor reconstruction quality. Hence I adopted a convolutional-based (CNN) architecture which preserves local spatial structures and extracts hierarchical features. 

Each convolutional layer is followed by batch normalisation to stabilise training and a LeakyReLU activation function. Instead of max pooling, I opted for stride-based downsampling to retain spatial information, as well as padding to prevent misalignment.

The decoder mirrors the encoder, using transpose convolutions for upsampling, though I refined kernel sizes and strides to mitigate checkerboard artifacts. Additionally, residual connections and self-attention in the bottleneck layer helped retain fine-grained details and capture long-range dependencies. 

## Hyperparameter Tuning

I made use of the Optuna library to automate the hyperparameter tuning process. By running trials, I systematically explored a complex search space of model configurations, including latent dimensions, network architecture, and loss function parameters. 

Optuna intelligently pruned underperforming configurations, ultimately helping me identify an optimal model design that balances compression efficiency and reconstruction quality. The hyperparameter tuning process allowed me to discover nuanced configurations that significantly improved the autoencoder's performance beyond what manual tuning could achieve.

## Training 

My model utilises the Lion optimiser, a momentum-based method with fewer parameters. This resulted in smoother convergence and better generalisation. It also reduced fluctuations in validation loss in later epochs.

I also implemented learning rate scheduling, initially using step decay but later switching to cosine annealing. This was helpful in preventing large oscillations in weight updates and therefore stabilising the training process. Without scheduling, I observed slower convergence and suboptimal reconstructions.

An issue that arose in the deeper layers of my model was gradient explosion - to combat this I incorporated gradient clipping, which ensured stable optimisation, especially at larger batch sizes.

## Results

[insert reconstructed images here]

## Visualisations

Here are some graphs (and other visualisations which I think look pretty cool) that I produced from training the model:

**Compression Quality Grid**
![Compression Quality Grid](https://github.com/ikemal12/autoencoder/blob/7983c6dd1ee7cdab7b26f93f7a8ae6cf41e77b8d/visualisations/compression_quality_grid_epoch_66.png)

**Compression Curve**
![Compression Curve](https://github.com/ikemal12/autoencoder/blob/7983c6dd1ee7cdab7b26f93f7a8ae6cf41e77b8d/visualisations/compression_curve_epoch_66.png)

**Training Curves**
![Training Curves](https://github.com/ikemal12/autoencoder/blob/7983c6dd1ee7cdab7b26f93f7a8ae6cf41e77b8d/visualisations/training_curves.png)

**Feature Maps**
![Feature Maps](https://github.com/ikemal12/autoencoder/blob/7983c6dd1ee7cdab7b26f93f7a8ae6cf41e77b8d/visualisations/feature_maps_epoch_61.png)

**t-SNE Visualization**
![t-SNE](https://github.com/ikemal12/autoencoder/blob/7983c6dd1ee7cdab7b26f93f7a8ae6cf41e77b8d/visualisations/tsne_epoch_66.png)

**Weight Distribution**
![Weight Distribution](https://github.com/ikemal12/autoencoder/blob/7983c6dd1ee7cdab7b26f93f7a8ae6cf41e77b8d/visualisations/weight_distribution.png)

**Parameter Counts**
![Parameter Counts](https://github.com/ikemal12/autoencoder/blob/7983c6dd1ee7cdab7b26f93f7a8ae6cf41e77b8d/visualisations/parameter_counts.png)

## Applications  

Apart from the main goal of this autoencoder which was to learn to perform image compression, I could have used it to learn compression for audio or video files just as well.

Furthermore there are a whole range of other applications I can tailor the autoencoder towards, such as:

* Dimensionality Reduction - e.g. for PCA algorithm or feature extraction for classification tasks.
* Anomaly Detection - e.g. fraud detection, network intrusions.
* Image Denoising - can adapt my current model to learn to remove grainy noise from old or blurry images.
* Image Generation - variational autoencoders (VAEs) can generate new realistic data.
* Super-Resolution - can adapt model to learn to reconstruct high-resolution images from low-resolution ones.
* and a lot more!

## Next Steps

While this autoencoder implementation performs quite well on the dataset I have trained it on, there is always room for improvement:

* Train the model on more images/larger dataset - this will be enable it to generalise better for unseen data.
* Train model on GPUs (Google Colab offers free tier usage) - this will permit for a more intense training scheme to be implemented, e.g. more epochs, layers, batch size etc.
