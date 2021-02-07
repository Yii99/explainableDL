import tensorflow as tf
import scipy as sp
import numpy as np


def resize_image(img, size, interpolation=0):
    '''

    :param img: image to be resized
    :param size: the target size of image
    :param interpolation: Interpolation between 0 (no interpolation) and 5 (maximum interpolation)
    :return:
    '''

    zoom_factors = np.asarray(size) / np.asarray(img.shape)
    return sp.ndimage.zoom(img, zoom_factors, order=interpolation)

def occlusion(model, image, target_class, size, stride, occlusion_value, resize=True):
    '''

    :param model: pre-trained model
    :param image: 3D input image
    :param target_class: the target output class for which to produce the heatmap
    :param size: the size of the occlusion patch
    :param stride: the stride defines the step to move the occlusion patch across the image
    :param occlusion_value: the value of the occlusion patch.
    :param resize: the output from the occlusion method is usually smaller than the original `image_tensor`.
                   If `True` (default), the output will be resized to fit the original shape (without interpolation).
    :return: A numpy array of the same shape as image_tensor, indicating the relevance of each image pixel.
    '''

    # 1. calculate the unoccluded probabilities
    img_tensor = tf.expand_dims(image, axis=-1)
    img_tensor = tf.expand_dims(img_tensor, axis=0)

    output = model(img_tensor, training=False)

    unoccluded_prob = output[:, :, :, :, target_class]

    # 2. generate occluded image
    depth = image.shape[0]
    width = image.shape[1]
    height = image.shape[2]

    zs = range(0, depth, stride)
    xs = range(0, width, stride)
    ys = range(0, height, stride)

    relevance_map = np.zeros((len(zs), len(xs), len(ys)))

    for i_x, x in enumerate(xs):
        x_from = max(x - int(size / 2), 0)
        x_to = min(x + int(size / 2), width)

        for i_y, y in enumerate(ys):
            y_from = max(y - int(size / 2), 0)
            y_to = min(y + int(size / 2), height)

            for i_z, z in enumerate(zs):
                z_from = max(z - int(size / 2), 0)
                z_to = min(z + int(size / 2), depth)

                image_tensor_occluded = image.copy()
                image_tensor_occluded[z_from:z_to, x_from:x_to, y_from:y_to] = occlusion_value

                # 3. calculate the occluded probabilities
                output = model(image_tensor_occluded[None], training=False)
                occluded_prob = output[:, :, :, :, target_class]

                # 4. calculate the difference of unoccluded and occluded probabilities
                diff = unoccluded_prob - occluded_prob
                diff = tf.squeeze(diff)

                # 5. calculate the global average of the difference and put the result at the corresponding position
                # to represents the relevance of this region to prediction
                relevance_map[i_z, i_x, i_y] = tf.math.reduce_mean(diff, axis=[0, 1, 2])

    # 6. filter out negative values
    relevance_map = np.maximum(relevance_map, 0)

    # 7. resize the relevance map and return
    if resize:
        relevance_map = resize_image(relevance_map, image.shape)

    return relevance_map

