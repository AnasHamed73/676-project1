# 676-project1

This project aims to outline the benefits of incorporating the following features to a conventional Deep Convolutional Generative Adversarial Network:

  * Label Conditioning: providing the label that represents each class of each training sample of the dataset along with each sample
	* Self Attention: injecting an attention layer in between the convolutional layers of both the generator and discriminator networks at specific locations
	* Wasserstein Loss: using a Wasserstein loss with a gradient penalty in lieu of a Binary Cross Entropy loss function
	* Spectral Normalization: used in addition to Batch Normaliztion, this normalization technique aims to stabilize the training process for both components of the network

