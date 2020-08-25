# Neural Style Transfer

Neural style transfer means composing one image in the style of another image (e.g. make your painting like Picasso or Van Gogh). This technique is outlined in <a href="https://arxiv.org/abs/1508.06576" class="external">A Neural Algorithm of Artistic Style</a> (Gatys et al.). 

The algorithm takes one content image to be tranfered and one style image to be based on as inputs.

<figure>
<img align="center" src="https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg" width="500"/>
<figcaption>input content image</figcaption>
</figure>

<figure>
<img align="center" src="https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg" alt="input style image" width="500"/>
<figcaption>input style image</figcaption>
</figure>

The network is usually a pre-trained convolutional neural network, from which we extract one layer as content layer and some layers as style layer. The content layer with the content image as input will represent content label, while the style layers with the style image as input will be processed (methods like gram_matrix) to style label. 

When training, the input content image goes through the network and returns content and style data, which are compared with the labels to generate one weighted loss. With gradient descent we can then modify the input content image to minimize the loss, trying to keep the content while change the style as close to the label as possible.

<figure>
<img src="https://tensorflow.org/tutorials/generative/images/stylized-image.png" width="500"/>
<figcaption>output transfered image</figcaption>
</figure>
