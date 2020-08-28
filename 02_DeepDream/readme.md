# Deep Dream

DeepDream is an experiment that visualizes the patterns learned by a neural network. Similar to when a child watches clouds and tries to interpret random shapes, DeepDream over-interprets and enhances the patterns it sees in an image. This practice is based on this [blog post](https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html) by Alexander Mordvintsev.

It does so by forwarding an image through the network, then calculating the gradient of the image with respect to the activations of a particular layer. The image is then modified to increase these activations (i.e. gradient ascent), enhancing the patterns of the layer seen by the network, and resulting in a dream-like image. This process was dubbed "Inceptionism" (a reference to InceptionNet, and the movie Inception).

Using different layers will result in different dream-like images. Deeper layers respond to higher-level features (such as eyes and faces), while earlier layers respond to simpler features (such as edges, shapes, and textures).

<figure>
<img src="https://www.tensorflow.org/tutorials/generative/images/dogception.png" alt="output image" width="500px"/>
<figcaption>dream-like image by enhancing intermediate layer features</figcaption>
</figure>
