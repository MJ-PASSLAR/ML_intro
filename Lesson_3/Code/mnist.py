# -*- coding: utf-8 -*-
"""MNIST.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/153hfSBb2VT6QL46AwHpvL1t3dQE6IcTz
"""

#Loading the MNIST dataset in Keras
from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print(train_images.shape)
print(train_labels.shape)

print(test_images.shape)
print(test_labels.shape)

#The network architecture
from keras import models
from keras import layers
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

network.summary()

#compile your netwotk
network.compile(optimizer='rmsprop',
loss='categorical_crossentropy',
metrics=['accuracy'])

#Preparing the image data
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

from tensorflow.keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#train your netwotk
history = network.fit(train_images, train_labels, epochs=20, batch_size=128)

#evaluate your network
test_loss, test_acc = network.evaluate(test_images, test_labels)
print(test_loss)

#visiualize data
history_dict = history.history
loss_values = history_dict['loss']
accuracy_values = history_dict['accuracy']

import matplotlib.pyplot as plt
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4,hspace=0.4)
plt.subplot(1,2,1)
plt.title('Training and validation loss')
plt.plot(loss_values)
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.subplot(1,2,2)
plt.title('Training and validation accuracy')
plt.plot(accuracy_values)
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()