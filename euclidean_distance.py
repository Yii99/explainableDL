import numpy as np
from DeepVisualization.get_model import get_model
import tensorflow as tf
from read_load_data import load_validation_data
from preprocessing.calculate_bounding_boxes import crop_to_same_size
from DeepVisualization.Vanilla_Backpropagation import Vanilla_Backpropagation
from DeepVisualization.Guided_Backpropagation import Guided_Backpropagation
from DeepVisualization.Deconvnet import DeconvNet
from DeepVisualization.GradCAM import GradCAM
from DeepVisualization.Guided_GradCAM import guided_gradcam
from DeepVisualization.Integrated_Gradients import get_integrated_gradients
from DeepVisualization.Smooth_Grad import sm_grad
from DeepVisualization.Occlusion import occlusion

def heatmap_distance(a, b):
    '''
    :param a: saliency map
    :param b: original mask with selected organ
    :return: the euclidean distance of a and b
    '''

    def preprocess(arr):
        # Preprocess an array for use in Euclidean distance
        arr = arr.flatten()
        # normalize to sum 1
        arr = arr / arr.sum()
        return arr

    a, b = preprocess(a), preprocess(b)

    # Euclidean distance.
    return np.sqrt(np.sum((a - b) ** 2))


def cal_dis(path, data_path, model_name, layer_name, class_of_interest, testing_files,
            VB, DN, GB, GC, GGC, IG, SM, guided_bp, vanilla, deconv, OC):
    '''

    :param path: the path where stored the model weights
    :param data_path: the path stored the medical images
    :param model_name: the model to be used
    :param layer_name: the last convolutional layer
    :param class_of_interest: the index of organ of interest
    :param testing_files: test dataset
    :param VB: if True, the euclidean distance of Vanilla-Backpropagation is calculated
    :param DN: if True, the euclidean distance of Guided-Backpropagation is calculated
    :param GB: if True, the euclidean distance of DeconvNet is calculated
    :param GC: if True, the euclidean distance of GradCAM is calculated
    :param GGC: if True, the euclidean distance of Guided-GradCAM is calculated
    :param IG: if True, the euclidean distance of Integrated Gradients is calculated
    :param SM: if True, the euclidean distance of Smooth Gradvis calculated
    :param guided_bp: if True, Smooth Grad is combined with Vanilla-Backpropagation
    :param vanilla: if True, Smooth Grad is combined with Guided-Backpropagation
    :param deconv: if True, Smooth Grad is combined with DeconvNet
    :param OC: if True, the euclidean distance of Occlusion is calculated
    :return: the averaged euclidean distance
    '''
    dis = []
    model = get_model(path, model_name)
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    n = 0
    for files in testing_files:
        n += 1
        print(n)
        image_array, mask_array, shape, spacing, origin = load_validation_data(data_path, files)
        zmin, zmax, xmin, xmax, ymin, ymax = crop_to_same_size(image_array)
        img_ori = image_array[zmin:zmax, xmin:xmax, ymin:ymax]
        mask_ori = mask_array[zmin:zmax, xmin:xmax, ymin:ymax]
        img = tf.expand_dims(img_ori, axis=0)
        img = tf.expand_dims(img, axis=-1)
        if VB:
            grads = Vanilla_Backpropagation(path, model_name, layer_name, class_of_interest, img_ori)
        elif DN:
            grads = DeconvNet(path, model_name, layer_name, class_of_interest, img_ori)
        elif GB:
            grads = Guided_Backpropagation(path, model_name, layer_name, class_of_interest, img_ori)
        elif GC:
            grads = GradCAM(grad_model, class_of_interest, img_ori)
        elif GGC:
            gb = Guided_Backpropagation(path, model_name, layer_name, class_of_interest, img_ori)
            gc = GradCAM(grad_model, class_of_interest, img_ori)
            grads = guided_gradcam(gc, gb)
        elif IG:
            grads = get_integrated_gradients(img, model, class_of_interest, baseline=None, num_steps=50).numpy()
            grads = (grads - grads.min()) / (grads.max() - grads.min())
        elif SM:
            grads = sm_grad(path, model_name, layer_name, class_of_interest, img_ori, 0.05, 50, guided_bp, vanilla, deconv)
        elif OC:
            grads = occlusion(model, img_ori, class_of_interest, 10, 5, 0, resize=True).numpy()
            grads = (grads - grads.min()) / (grads.max() - grads.min())

        dis1 = heatmap_distance(grads, mask_ori == class_of_interest)
        dis.append(dis1)
        dis_avg = np.mean(dis)
    return dis_avg
