# CNN-Based-Hindi-Handwritten-Character-Recognition-and-Speech-Conversion

## Description
This is a Character Recognition System which I developed for Devanagari Script. The learning model was trained on 92,000 Grayscale images (32x32 pixels) of 46 characters, digits 0 to 9 and consonants “ka” to “gya”. Out of these 92,000 images, 78,200 images is for training and 13,800 images is for testing. The optimal result, 98.7% accuracy was obtained using Convolutional Neural Network.

## Code Requirements
You can install Anaconda for python which resolves all the dependencies for machine learning.
After installation of anaconda go with the jupyter notebook and open sarthak.ipynb file.

## Trained on system having following configurations
* GPU-Nvidia Tesla P100 based AI/Deep Learning Server (2x16GB) 
* RAM - 64 GB
You can also go with your laptop gpu and cpu for training

## Architecture
### CONV2D(32,3,3)-->CONV2D(64,3,3)-->Maxpooling2D(2,2)-->CONV2D(128,3,3)-->CONV2D(256,3,3)-->Maxpooling(2,2)-->FLATTEN-->RELU-->SOFTMAX

## Results
Validation Accuracy--> 98.7%

## Required Python modules
* Tensorflow-GPU
* Keras
* Numpy
* OpenCV
* Pyttsx
* Matplotlib(optional)

## Files Information
* Sarthak.ipynb contains the codes which is written in python language
* madel.h5 is a trained model of 20 epochs

## Steps to run this project
* Download the datasets which is available in UCI Machine Learning   Repository(https://archive.ics.uci.edu/ml/datasets/Devanagari+Handwritten+Character+Dataset) 
* The dataset folders contains two subfolders (train and test)
* Install all the python modules in your system 
* Download the above attached files in one folder
* Open Sarthak.ipynb file using jupyter notebook 
* Copy the filepath of the train folder and paste it in the Sarthak.ipynb file In[4] block to get the images from your directory do the     same with test folder also
*  Run all the blocks in Sarthak.ipynb file using shift + enter.
* Block In[8] in sarthak.ipynb is for the preprocessing and prediction 

## Additional Information
* You can go for more number of layers. 
* Add regulizers to prevent overfitting.
* You can also add your own images in the datasets for better accuracy.

## Collaborators
1. Sarthak Sinha(Student, ECE, AIT)
2. Mohammad Faizal(Student, ECE, AIT)
3. Mrs. Kruthika KR (Asst. Professor, ECE, AIT)
4. Mr Sandeep Kumar (Asst. Professor, ECE, AIT)
