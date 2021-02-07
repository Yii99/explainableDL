import tensorflow as tf
from DeepVisualization.get_model import get_model

def Guided_Backpropagation(path, model_name, layer_name_BP, class_of_interest, input):
    '''
    :param path: the path where stored the model weights
    :param model_name: the model to be used
    :param layer_name_BP: the last convolutional layer
    :param class_of_interest: the number of organ of interest
    :param input: input image of model
    :return: normalized Gradients
    '''

    # 1. define the new rule of ReLU
    @tf.custom_gradient
    def guidedRelu(x):
        def grad(dy):
            return tf.cast(dy>0,"float32") * tf.cast(x>0, "float32") * dy
        return tf.nn.relu(x), grad

    # 2. load the pre-trained model
    model = get_model(path, model_name)
    gb_model = tf.keras.models.Model(
                                    [model.inputs],
                                    [model.get_layer(layer_name_BP).output, model.output])

    # 3.apply the new rule of ReLU to each ReLU layer
    layer_dict = [layer for layer in gb_model.layers[1:] if hasattr(layer,'activation')]
    for layer in layer_dict:
        if layer.activation == tf.keras.activations.relu:
           layer.activation = guidedRelu

    # 4. calculate the gradients
    with tf.GradientTape() as tape:
         inputs = tf.cast(input, tf.float32)
         tape.watch(inputs)
         feature_map, logits = gb_model(inputs, training=False)
         loss = feature_map[:, :, :, :, class_of_interest]

    grads = tape.gradient(loss, inputs)[0]
    grads = tf.squeeze(grads).numpy()
    grads = (grads - grads.min()) / (grads.max() - grads.min())

    return grads
