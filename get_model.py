from models.metrics import tversky_loss, tversky_coefficient, tversky_coefficient_class0, tversky_coefficient_class1, \
    tversky_coefficient_class2, tversky_coefficient_class3, focal_tversky_loss, dice_coefficient, dice_loss, \
    dice_coefficient_class0, dice_coefficient_class1, dice_coefficient_class2, dice_coefficient_class3
import tensorflow as tf

def get_model(path, model_name):
    '''

    :param path: the path where stored the model weights
    :param model_name: the model to be used
    :return: the weights of pre-trained model
    '''

    MODEL_PATH = path + model_name + '/model_best.h5'
    print('Loading The Model from this path--{}'.format(MODEL_PATH))
    model = tf.keras.models.load_model(MODEL_PATH,
                                       custom_objects={
                                           'tversky_loss': tversky_loss,
                                           'tversky_coefficient': tversky_coefficient,
                                           'tversky_coefficient_class0': tversky_coefficient_class0,
                                           'tversky_coefficient_class1': tversky_coefficient_class1,
                                           'tversky_coefficient_class2': tversky_coefficient_class2,
                                           'tversky_coefficient_class3': tversky_coefficient_class3,
                                           'tf': tf,
                                           'GlorotUniform': tf.keras.initializers.glorot_uniform,
                                           'focal_tversky_loss': focal_tversky_loss,
                                           'dice_coefficient': dice_coefficient,
                                           'dice_loss': dice_loss,
                                           'dice_coefficient_class0': dice_coefficient_class0,
                                           'dice_coefficient_class1': dice_coefficient_class1,
                                           'dice_coefficient_class2': dice_coefficient_class2,
                                           'dice_coefficient_class3': dice_coefficient_class3,

                                       })

    return model