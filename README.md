# ECGR-4127-Machine_Learning_for_IoT
Compilation of homeworks and projects from ECGR-4127 Machine Learning for IoT

## Homework 1:
Small python script to refresh coding abilities for the course. 

## Homework 2:
Built 3 CNNs to use on the CIFAR-10 dataset. The first model just had convolutional layers. The second had depthwise separable convolutional layers. The third model added dropout layers to the same structure of the first model. The last model was the best model I could come up with (validation accuracy-wise) with no more than 50,000 parameters. A spreadsheet is included where we performed calculations to find the number of parameters and MACs for each layer in the first two models.

## Homework 3:
Assignment to demonstrate how to set up a model and train it on an Arduino TinyML board. Converted a Keras model into a TFLM model. Instantiated the model, including all necessary components, and got it running on the Arduino TinyML board. Added the ability to measure and display the time required for printing a single statement across to a terminal across the USB and running a single inference of the model. 

## Homework 4:
Wrote a python script to analyze the outcomes vs predictions of a dataset, precision, recall and ROC curve. Made a dataset of 10 images (some of dogs and some of not dogs) and trained a model to detect the dogs. Wrote a script to evaluate the model's accuracy.

## Project 1:
Used the CIFAR-10 dataset to train a CNN to recognize images of frogs. Used the Arduino IDE to deploy the model to an Arduino Nano33 BLE board. The light on the board turns green when a frog is detected in front of the camera.

## Project 2:
Used 25 .wav files of the word "banana" and .wav files of the word "yes" from the Google 30 dataset to train a machine learning model to recognize and classify a spoken word as "banana", "yes", silence, or unknown. Used methods of data augmentation such as pitch and time shifting to generate more .wav files to train the model on. Created a CNN and LSTM model to compare their results. Deployed the models to an Arduino Nano33 BLE board. The light on the board lit up green when the word "banana" or "yes" was recognized.
