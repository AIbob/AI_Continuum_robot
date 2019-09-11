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

index = np.arange(0,X_data.shape[0],1)
random.shuffle(index)

X_rand = np.mat(X_data[index,:])
Y_rand = np.mat(Y_data[index,:])

x_train = np.mat(np.zeros(X_rand.shape))
y_train = np.mat(np.zeros(Y_rand.shape))

x_train[:,0] = (X_rand[:,0] - np.min(X_data[:,0]))/(np.max(X_data[:,0]) - np.min(X_data[:,0]))
x_train[:,1] = (X_rand[:,1] - np.min(X_data[:,1]))/(np.max(X_data[:,1]) - np.min(X_data[:,1]))

y_train[:,0] = (Y_rand[:,0] - np.min(Y_data[:,0]))/(np.max(Y_data[:,0]) - np.min(Y_data[:,0]))
y_train[:,1] = (Y_rand[:,1] - np.min(Y_data[:,1]))/(np.max(Y_data[:,1]) - np.min(Y_data[:,1]))
y_train[:,2] = (Y_rand[:,2] - np.min(Y_data[:,2]))/(np.max(Y_data[:,2]) - np.min(Y_data[:,2]))

#define neual network
model = keras.Sequential([
    keras.layers.Dense(10, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.relu),
    keras.layers.Dense(2,activation=tf.nn.relu)
])

model.compile(optimizer='sgd', 
              loss='mean_squared_error',
              metrics=['accuracy'])

model.fit(y_train,x_train, epochs = 30)

test_data = y_train[range(0,390000,1000),:] 
test_x = x_train[range(0,390000,1000),:]
pre_out = model.predict(test_data)

x_test = np.zeros(test_x.shape)

x_test[:,0] = pre_out[:,0] * (np.max(X_data[:,0]) - np.min(X_data[:,0])) + np.min(X_data[:,0])
x_test[:,1] = pre_out[:,1] * (np.max(X_data[:,1]) - np.min(X_data[:,1])) + np.min(X_data[:,1])

test_x[:,0] = test_x[:,0] * (np.max(X_data[:,0]) - np.min(X_data[:,0])) + np.min(X_data[:,0])
test_x[:,1] = test_x[:,1] * (np.max(X_data[:,1]) - np.min(X_data[:,1])) + np.min(X_data[:,1])

err = np.mat(x_test) - np.mat(test_x)
fig = plt.figure()
plt.plot(err[:,0],color='r',linestyle='-')
plt.plot(err[:,1],color='b',linestyle='-')
plt.show()