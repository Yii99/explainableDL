import tensorflow as tf
import os
import numpy as np
from skimage.transform import resize


def GradCAM(grad_model, class_of_interest, img):
    '''
    :param grad_model: load the pre-trained model and get the feature map generated by the last convolutional layer and the output of model
    :param class_of_interest: the number of organ we want to show
    :param img: input image of model
    :return: the normalized heatmap calculated by this method
    '''

    # 1. calculate the gradients of the score of class with respect to the feature maps generated by the last convolutional layer.
    with tf.GradientTape() as tape:
        conv_outputs, logits = grad_model(img, training=False)
        loss = logits[:, :, :, :, class_of_interest]
    output = conv_outputs[0]
    Grad_gradients = tape.gradient(loss, conv_outputs)[0]

    # 2. calculate alpha by using global average pooling
    weights = tf.math.reduce_mean(Grad_gradients, axis=[0, 1, 2])

    # 3. calculate the weighted sum of all feature maps
    cam = np.zeros(output.shape[0:3], dtype=np.float32)

    for index, w in enumerate(weights):
        cam += w * output[:, :, :, index]

    # 4. resize cam to the same size as the original image
    capi = resize(cam, (192, 64, 64))

    # 5. normalization
    capi = np.maximum(capi, 0)
    heatmap = (capi - capi.min()) / (capi.max() - capi.min())

    return heatmap

