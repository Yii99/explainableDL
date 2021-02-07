import matplotlib.pyplot as plt
import tensorflow as tf

def plot_with_mask(ori_image, inputs, cmap, figpath, name, titles, overlay, overlay_alpha, verbose, subtitle, axial_slice_count, coronal_slice_count, sagittal_slice_count):
    '''

    :param ori_image: the original 3D medical image
    :param inputs: a list of the saliency maps we want to visualize and compare
    :param cmap: define the color of the generated figure
    :param figpath: the path to store the generated figure
    :param name: the name of the generated figure to be saved
    :param titles: a list of title to each saliency map we want to visualize
    :param overlay: if True, plot the overlay of each saliency map with the original image
    :param overlay_alpha: define the degree of transparency
    :param verbose: if True, plot the overlay of the original image and the selected saliency map at the last column of figures
    :param subtitle: is the title of the generated figure
    :param axial_slice_count: the index of slice we want to show in axial direction
    :param coronal_slice_count: the index of slice we want to show in coronal direction
    :param sagittal_slice_count: the index of slice we want to show in sagital direction
    '''
    axial_imgs = []
    coronal_imgs = []
    sagittal_imgs = []
    for img in inputs:
        img = tf.image.rot90(img, 2)
        axial_img = tf.squeeze(img[axial_slice_count, :, :])
        coronal_img = tf.squeeze(img[:, coronal_slice_count, :])
        sagittal_img = tf.squeeze(img[:, :, sagittal_slice_count])
        axial_imgs.append(axial_img)
        coronal_imgs.append(coronal_img)
        sagittal_imgs.append(sagittal_img)

    if overlay:
        fig, axarr = plt.subplots(3, len(inputs), figsize=(8, 8))
        fig.suptitle(subtitle, fontsize=12)
        axial_ori_img = tf.squeeze(ori_image[axial_slice_count, :, :])
        coronal_ori_img = tf.squeeze(ori_image[:, coronal_slice_count, :])
        sagittal_ori_img = tf.squeeze(ori_image[:, :, sagittal_slice_count])
        for i, img in enumerate(axial_imgs):
            axarr[0, i].imshow(axial_imgs[i], cmap=cmap)
            axarr[0, i].imshow(axial_ori_img, alpha=overlay_alpha)
            axarr[0, i].axis('off')
            axarr[0, i].set_title((titles[i]), fontsize=(10))
        for i, img in enumerate(coronal_imgs):
            axarr[1, i].imshow(coronal_imgs[i], cmap=cmap)
            axarr[1, i].imshow(coronal_ori_img, alpha=overlay_alpha)
            axarr[1, i].axis('off')
            axarr[1, i].set_title((titles[i]), fontsize=(10))

        for i, img in enumerate(sagittal_imgs):
            axarr[2, i].imshow(sagittal_imgs[i], cmap=cmap)
            axarr[2, i].imshow(sagittal_ori_img, alpha=overlay_alpha)
            axarr[2, i].axis('off')
            axarr[2, i].set_title((titles[i]), fontsize=(10))


    else:
        if verbose:
           fig, axarr = plt.subplots(3, len(inputs) + 1, figsize=(8, 8))
           fig.suptitle(subtitle, fontsize=12)
           axarr[0, len(inputs)].imshow(axial_imgs[-1], cmap=cmap)
           axarr[0, len(inputs)].imshow(axial_imgs[0], alpha=overlay_alpha)
           axarr[0, len(inputs)].axis('off')
           axarr[0, len(inputs)].set_title('Overlay', fontsize=(10))

           axarr[1, len(inputs)].imshow(coronal_imgs[-1], cmap=cmap)
           axarr[1, len(inputs)].imshow(coronal_imgs[0], alpha=overlay_alpha)
           axarr[1, len(inputs)].axis('off')
           axarr[1, len(inputs)].set_title('Overlay', fontsize=(10))

           axarr[2, len(inputs)].imshow(sagittal_imgs[-1], cmap=cmap)
           axarr[2, len(inputs)].imshow(sagittal_imgs[0], alpha=overlay_alpha)
           axarr[2, len(inputs)].axis('off')
           axarr[2, len(inputs)].set_title('Overlay', fontsize=(10))
        else:
            fig, axarr = plt.subplots(3, len(inputs), figsize=(8, 8))
            fig.suptitle(subtitle, fontsize=8)

        for i, img in enumerate(axial_imgs):
            axarr[0, i].imshow(axial_imgs[i], cmap=cmap)
            axarr[0, i].axis('off')
            axarr[0, i].set_title((titles[i]), fontsize=(10))

        for i, img in enumerate(coronal_imgs):
            axarr[1, i].imshow(coronal_imgs[i], cmap=cmap)
            axarr[1, i].axis('off')
            axarr[1, i].set_title((titles[i]), fontsize=(10))

        for i, img in enumerate(sagittal_imgs):
            axarr[2, i].imshow(sagittal_imgs[i], cmap=cmap)
            axarr[2, i].axis('off')
            axarr[2, i].set_title((titles[i]), fontsize=(10))


    fig.savefig(figpath + name)