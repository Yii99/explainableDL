import numpy as np
import tensorflow as tf
from DeepVisualization.Vanilla_Backpropagation import Vanilla_Backpropagation
from DeepVisualization.Deconvnet import DeconvNet
from DeepVisualization.Guided_Backpropagation import Guided_Backpropagation

def sm_grad(path, model_name, layer_name, class_of_interest, x_value, stdev_spread, nsamples, guided_bp, vanilla, deconv):
    '''

    :param path: the path where stored the model weights
    :param model_name: the model to be used
    :param layer_name: the layer, which we want to use its output to calculate gradients
    :param class_of_interest: the index of organ we want to show
    :param x_value: the original input image
    :param stdev_spread: standard deviation of gaussian noise
    :param nsamples: number of samples (n gaussian noise)
    :param guided_bp: if True, guided-backpropagation is combined with smooth grad.
    :param vanilla: if True, vanilla-backpropagation is combined with smooth grad.
    :param deconv: if True, deconvnet is combined with smooth grad.
    :return:
    '''

    # 1. normalize deviation
    stdev = stdev_spread * (np.max(x_value) - np.min(x_value))
    total_gradients = np.zeros_like(x_value)

    # 2. generate n random gaussian noise and the image with noises
    for i in range(nsamples):
      print("Iteration {} of 50".format(i))
      noise = np.random.normal(0, stdev, x_value.shape)
      x_plus_noise = x_value + noise
      x_plus_noise = tf.expand_dims(x_plus_noise, axis=0)
      x_plus_noise = tf.expand_dims(x_plus_noise, axis=-1)

      # 3. use the image with noises as the input to each selected method
      if guided_bp:
         grad = Guided_Backpropagation(path, model_name, layer_name, class_of_interest, x_plus_noise)
      elif vanilla:
           grad = Vanilla_Backpropagation(path, model_name, layer_name, class_of_interest, x_plus_noise)
      elif deconv:
           grad = DeconvNet(path, model_name, layer_name, class_of_interest, x_plus_noise)
      total_gradients = tf.squeeze(total_gradients)

      # 4. gradients summation
      total_gradients += grad

    # 5. calculate the average of gradients and return
    return total_gradients / nsamples