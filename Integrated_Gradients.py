import tensorflow as tf
import numpy as np

def compute_gradients(model, img_input, target_class_idx):

    '''
    :param model:
    :param img_input: the original input image of model
    :param target_class_idx: the index of organ we want to show
    :return: the gradients of the score of class with respect to the input image
    '''

    images = tf.cast(img_input, tf.float32)

    with tf.GradientTape() as tape:
         tape.watch(images)
         conv_outputs = model(images)
         outputs = conv_outputs[:, :, :, :, target_class_idx]
    gradients = tape.gradient(outputs, images)

    return gradients

def get_integrated_gradients(img_input, model, target_class_idx, baseline=None, num_steps=50):
    '''

    :param img_input: the original input image of model
    :param model: the weights of pre-trained model
    :param target_class_idx: the index of organ we want to show
    :param baseline: the baseline image to start with for interpolation
    :param num_steps: number of interpolation steps between the baseline
            and the input used in the computation of integrated gradients. These
            steps along determine the integral approximation error. By default,
            num_steps is set to 50.
    :return: integrated gradients w.r.t input image
    '''

    # If baseline is not provided, start with a black image, which has the same size as the input image.
    img_size= img_input.shape
    if baseline is None:
        baseline = np.zeros(img_size).astype(np.float32)
    else:
        baseline = baseline

    # 1. generate interpolated inputs between baseline and input.
    interpolated_image = [
        baseline + (step / num_steps) * (img_input - baseline)
        for step in range(num_steps + 1)
    ]
    interpolated_image = np.array(interpolated_image).astype(np.float32)
    print('step1')

    # 2. compute gradients between model outputs and interpolated inputs.
    grads = []
    for i, img in enumerate(interpolated_image):
        grad = compute_gradients(model, img, target_class_idx)
        grads.append(grad[0])
    grads = tf.convert_to_tensor(grads, dtype=tf.float32)
    print('step2')
    # 3. integral approximation through averaging gradients.
    grads = (grads[:-1] + grads[1:]) / 2.0
    avg_grads = tf.math.reduce_mean(grads, axis=0)
    print('step3')
    # 4. scale integrated gradients with respect to input
    integrated_grads = (img_input - baseline) * avg_grads
    print('step4')
    integrated_grads = tf.squeeze(integrated_grads)
    return integrated_grads