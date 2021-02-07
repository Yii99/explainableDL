# explainableDL
**Due to the confidentiality regulations, only part of the code can be displayed.**
## Dataset
3D PET images
## Input Pipeline
![accuracy](https://github.com/Yii99/explainableDL/blob/main/fig/flow.png)
### Packages 
* Tensorflow 2.0
* NumPy v.1.17.3
* SimpleITK v.1.2.3
* matplotlib v.3.2.1
* scipy v.1.4.1
### Preprocessing
#### Data Preparation
* use SITK library  to read the series of image files into a SimpleITK image
* integrate all three separate organ masks into a mask, where the liver-array, the spleen-array and the vertebra-array are labeled respectively by integer 1, 2 and 3.
#### TF-Record
* run prep_data to transform the data set into tfrecords.
#### Bounding Box
* run calculate_bounding_boxes to crop images as organs normally lie in same range of image
#### Image Cropping
* run crop_to_same_size to crop the image to the target size
#### Image Denoising
* run image_denoising to denoise the image
### Model Establishing
![accuracy](https://github.com/Yii99/explainableDL/blob/main/fig/Att-Unet.pdf)
### Training
run training.py, set the following parameters:
* path: the directory where the code lies
* data_path: the directory where the data set (in tfrecord format) lies
* model_name: the name of the folder where you want the trained model to be saved in
* model_type: which type of DNN architecture you want to train. Attention U-Net ('attunet')
* loss_type: which type of loss type you want the architecture to train with. Choose from Tversky Loss ('tversky'), Dice Loss ('dice') or Focal Tversky Loss ('focal')
* epoch_num: the number of epochs you want your model to train
* img_size: the resolution of the data set. '0.5' for half-size resolution.
* segmentation: the organs that your model shall segment. Choose from 'all' for simultaneous segmentation of liver, spleen and vertebra, 'liver' for liver segmentation only, 'spleen' for spleen segmentation only and 'vertebra' for vertebra segmentation only.
### Testing
run testing.py, set the following parameters:
* path: the directory where the code lies
* data_path: the directory where the data set (in tfrecord format) lies
* model_name: the name of the folder where you want the trained model to be saved in
* result_folder: the name of the folder where you want the predictions to be saved in
* img_size: the resolution of the data set. '0.5' for half-size resolution
* segmentation: the organs that your model shall segment. Choose from 'all' for simultaneous segmentation of liver, spleen and vertebra, 'liver' for liver segmentation only, 'spleen' for spleen segmentation only and 'vertebra' for vertebra segmentation only.
### Evaluation
run result_evaluation.py, set the following parameters:
* model_name: the name of the folder where you want the trained model to be saved in
* result_folder: the name of the folder where you want the predictions to be saved in
* num_classes: the number of classes which the model predicted. Choose between 4 for all organs + background and 2 for single organ + background.
### Full Pipeline
run full pipeline to complete directly all steps and get the result.
## Result
### Training Performance
#### Loss
![accuracy](https://github.com/Yii99/explainableDL/blob/main/fig/loss.png)
#### Dice coefficient
![accuracy](https://github.com/Yii99/explainableDL/blob/main/fig/dc.png)
### Evaluation
![accuracy](https://github.com/Yii99/explainableDL/blob/main/fig/tcdc.png)
![accuracy](https://github.com/Yii99/explainableDL/blob/main/fig/cm.png)
### Explainable Results 
Here, we only show an example of our results (the saliency map of the liver)
![accuracy](https://github.com/Yii99/explainableDL/blob/main/fig/total_1.png)
