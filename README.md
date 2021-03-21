# emotion-detection

Emotion detection is allowing machine to understand the feelings in the human face. This is done by Convolutinal Neural Networks 
which is a powerful tool in the Computer Vision environment. In the below you can find the steps that I follow.


## 1) Dataset selection

It is difficult to create a dataset for faces , that is why fer2013 dataset is used.

## 2) Preprocessing

Pixels are the important players in our case, so normalization and one hot encoding techniques are used to help model understand
better.

## 3) HyperParameter tunning

It is crucial to find best paremeters for our CNN model. How many layers will we use?, What is the kernel size of convolutional layers?
Which activation function should we choose? ....


## 4) Model training

I apply 150 epochs and early stopping technique to avoid overfit my model.

## 5) Extract json and h5 files

These files contain the weights of the our model. From now on, we can use our model in everywhere.
This is the final operation in the google colab side. Now we will take our model and use it in VS Code.


## 6) Camera Code

In the VS Code, we get our model from h5 file and also get pretrained face classifier from haarcascade_frontalface_default.xml, which is
famous xml file for face detection. Firstly face is detected and region of interest of the face is taken. Secondly, this image is converted 
to array for prediction operation. This input(array) is given to our model and the highest probability emotion is chosen.

