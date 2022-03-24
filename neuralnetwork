import os
import cv2
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf 


# loading MNIST data
mnist = tf.keras.datasets.mnist

# load dataset 
(trainX, trainY), (testX, testY) = mnist.load_data()

trainX = tf.keras.utils.normalize(trainX , axis = 1)
testX = tf.keras.utils.normalize(testX , axis = 1)

# creating the model 
model = tf.keras.models.Sequential() 

# add layers to model
model.add(tf.keras.layers.Flatten(input_shape = (28, 28))) # turns grid into 28x28 = 784 pixels line
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu)) # 'rectify linear unit'
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu)) # 'rectify linear unit'
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu)) # 'rectify linear unit'
model.add(tf.keras.layers.Dense(128, activation = tf.nn.elu)) # 'linear unit'
model.add(tf.keras.layers.Dropout(0.2)) # 
model.add(tf.keras.layers.Dense(15, activation = tf.nn.softmax)) # output layer, each neuron is gonna have 10 digits and will add up to a value between 0 and 1
# probability of each digit to be the right answer

# compile model
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

model.fit(trainX, trainY, epochs = 20)

#save model 
model.save('handwrittenmodel.h5')

model = tf.keras.models.load_model('handwrittenmodel.h5')

loss, accuracy = model.evaluate(testX, testY)

print(loss)
print(accuracy)


