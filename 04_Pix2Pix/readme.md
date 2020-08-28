# Pixelwise image to image translation with conditional GANs

Conditional Generative Adversarial Networks introduce additional information to training process, such as labels, tags or in theory any other information. In image generation, this allows us to learn features from input images, i.e. train the image-target pairs to do style transfer.

In this image to image translation, the inputs are building facade architure drafts from which we want to generate real-like facades. Each training sample is an image pair.

The generator is a U-Net architure which takes the draft as input and output a predicted facade image with the same size as target image.

The discriminator is a PatchGAN architure (or simply a CNN) which takes the input image and target image pair as input, and returns a (30,30,1) matrix.

The generator loss is a sigmoid cross entropy loss of the generated images and an array of ones. To allow the generated image become structurally similar to the target image, we include L1 loss which is MAE (mean absolute error) between the generated image and the target image. The total loss is equal to ```gan_loss + LAMBDA * l1_loss```.

![Generator Update Image](https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/images/gen.png?raw=1)

The discriminator loss has two parts: real_loss is a sigmoid cross entropy loss of the real images and an array of ones(since these are the real images); generated_loss is a sigmoid cross entropy loss of the generated images and an array of zeros(since these are the fake images). Then the total_loss is the sum of real_loss and the generated_loss.

![Discriminator Update Image](https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/images/dis.png?raw=1)

When training, for each example input generate an output. The discriminator receives the input_image and the generated image as the first input and get discriminator-real-output. The second input is the input_image and the target_image to get the discriminator-gen-output. Next, we calculate the generator and the discriminator loss. Then, we calculate the gradients of loss with respect to both the generator and the discriminator variables(inputs) and apply those to the optimizer.

The final result looks like below.

![sample output_1](https://www.tensorflow.org/images/gan/pix2pix_1.png)
![sample output_2](https://www.tensorflow.org/images/gan/pix2pix_2.png)



