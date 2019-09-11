from __future__ import absolute_import, division, print_function, unicode_literals

# 导入TensorFlow和tf.keras
import tensorflow as tf
from tensorflow import keras

# 导入辅助库
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

import random


inputdata = np.load('input.npy')
outputdata = np.load('output.npy')

X_data = np.mat(inputdata.T)
Y_data = np.mat(outputdata.T)

x_train = np.mat(np.zeros(X_data.shape))
y_train = np.mat(np.zeros(Y_data.shape))
index = np.arange(0,x_train.shape[0],1)
random.shuffle(index)

X_rand = np.mat(X_data[index,:])
Y_rand = np.mat(Y_data[index,:])

x_train[:,0] = (X_rand[:,0] - np.min(X_data[:,0]))/(np.max(X_data[:,0]) - np.min(X_data[:,0]))
x_train[:,1] = (X_rand[:,1] - np.min(X_data[:,1]))/(np.max(X_data[:,1]) - np.min(X_data[:,1]))

y_train[:,0] = (Y_rand[:,0] - np.min(Y_data[:,0]))/(np.max(Y_data[:,0]) - np.min(Y_data[:,0]))
y_train[:,1] = (Y_rand[:,1] - np.min(Y_data[:,1]))/(np.max(Y_data[:,1]) - np.min(Y_data[:,1]))
y_train[:,2] = (Y_rand[:,2] - np.min(Y_data[:,2]))/(np.max(Y_data[:,2]) - np.min(Y_data[:,2]))

model = keras.Sequential([
    keras.layers.Dense(30, activation=tf.nn.relu),
    keras.layers.Dense(3, activation=tf.nn.relu)
])

model.compile(optimizer='sgd', 
              loss='mean_squared_error',
              metrics=['accuracy'])

model.fit(x_train,y_train, epochs = 30)

test_data = x_train[range(0,390000,1000),:] 
test_y = y_train[range(0,390000,1000),:]
pre_out = model.predict(test_data)

y_test = np.zeros(test_y.shape)
y_test[:,0] = pre_out[:,0] * (np.max(Y_data[:,0]) - np.min(Y_data[:,0])) + np.min(Y_data[:,0])
y_test[:,1] = pre_out[:,1] * (np.max(Y_data[:,1]) - np.min(Y_data[:,1])) + np.min(Y_data[:,1])
y_test[:,2] = pre_out[:,2] * (np.max(Y_data[:,2]) - np.min(Y_data[:,2])) + np.min(Y_data[:,2])

test_y[:,0] = test_y[:,0] * (np.max(Y_data[:,0]) - np.min(Y_data[:,0])) + np.min(Y_data[:,0])
test_y[:,1] = test_y[:,1] * (np.max(Y_data[:,1]) - np.min(Y_data[:,1])) + np.min(Y_data[:,1])
test_y[:,2] = test_y[:,2] * (np.max(Y_data[:,2]) - np.min(Y_data[:,2])) + np.min(Y_data[:,2])

err = np.mat(y_test)-np.mat(test_y)

fig = plt.figure()
plt.plot(err[:,0].tolist(), color='b',linestyle='-')
plt.plot(err[:,1].tolist(),color='r')
plt.plot(err[:,2].tolist(),color='y')
plt.show()