# GAN-in-Practice

This repository is a collection of tutorials and pratices related to Generative Adversarial Networks. 

***01 to 07 are based on [tenforflow official generative tutorials](https://www.tensorflow.org/tutorials/generative/style_transfer).***

**01: [Neural Style Transfer](./01_Neural_Style_Transfer/)** --- transfer an image to a certain style based on the paper [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576).

<figure>
<img src="https://tensorflow.org/tutorials/generative/images/stylized-image.png" style="width: 500px;"/>
</figure>

**02: [Deep Dream](./02_DeepDream/)** --- transfer an image to dream-like style via exciting intermediate features based on the blog [Inceptionism: Going Deeper into Neural Networks](https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html).

<figure>
<img src="https://www.tensorflow.org/tutorials/generative/images/dogception.png"  width="500px"/>
</figure>

**03: [DCGAN](./03_DCGAN/)** --- train a [generative adversarial network](https://arxiv.org/pdf/1511.06434.pdf) to generate handwriting numbers.

<figure>
<img src="https://tensorflow.org/images/gan/dcgan.gif" width="500px"/>
</figure>

**04: [Pix2Pix](./04_Pix2Pix/)** --- Pixelwise image to image translation with conditional GANs, as described in [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004).

![sample output_1](https://www.tensorflow.org/images/gan/pix2pix_1.png)
![sample output_2](https://www.tensorflow.org/images/gan/pix2pix_2.png)

**05: [CycleGAN](./05_CycleGAN/)** --- image style transfer (without paired inputs) based on [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/pdf/1703.10593.pdf).

![Output Image 1](https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/images/horse2zebra_1.png?raw=1)
![Output Image 2](https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/images/horse2zebra_2.png?raw=1)

**06: [FGSM](./06_FGSM/)** --- learn to fool a neural network using [Fast Gradient Signed Method](https://arxiv.org/abs/1412.6572).

![Adversarial Example](https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/images/adversarial_example.png?raw=1)

**07: [AutoEncoder](./07_AutoEncoder/)** --- Latent space based image reconstruction and generation with autoencoder and [variational autoencoder](https://arxiv.org/abs/1906.02691). (addition: take a look at [conditional VAE](https://towardsdatascience.com/understanding-conditional-variational-autoencoders-cd62b4f57bf8))

![CVAE image latent space](https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/images/cvae_latent_space.jpg?raw=1)

**08: [U-GAT-IT](./08_UGATIT/)** --- improved unpaired image style transfer based on [**Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation (ICLR 2020)**](https://arxiv.org/abs/1907.10830).

![U-GAT-IT example](./08_UGATIT/assets/teaser.png)


