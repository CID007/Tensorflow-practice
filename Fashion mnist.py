import pylab as pl
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
print(tf.__version__)

# get train and test data. Data has already been sorted in train and test

(train_data, train_labels), (test_data, test_labels) = fashion_mnist.load_data()

# Show the first training example
print(f"Training sample:\n{train_data[0]}\n")
print(f"Training Label: {train_labels[0]}")

#check shape of data
print(train_data.shape)
print(train_labels.shape)
print(test_data.shape)
print(test_labels.shape)

# Check shape of single example
print(train_data[0].shape)
print(train_labels[0].shape)

import matplotlib.pyplot as plt
plt.imshow(train_data[7])
plt.show()
print(train_labels[7])

# Create a dataset with classes names

class_names = ["T-shir/top", "trouser", "Pullover", "Dress", "Coat", "Sandals", "Shirt", "Sneaker", "Bag", "Ankle boot"]

print(len(class_names))

# Plot an example image and its label

plt.imshow(train_data[17], cmap=plt.cm.binary)
plt.title(class_names[train_labels[17]])
plt.show()

# Plot multiple random of fashion mnist
import random
plt.figure(figsize=(7, 7))
for i in range(4):
    ax = plt.subplot(2, 2, i+1)
    rand_index = random.choice(range(len(train_data)))
    plt.imshow(train_data[rand_index], cmap=plt.cm.binary)
    plt.title(class_names[train_labels[rand_index]])
    plt.axis(False)
    plt.show()

# Build model
# Sset seed
tf.random.set_seed(42)

# create the model
model_11 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model_11.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                 optimizer=tf.keras.optimizers.Adam(),
                 metrics=['accuracy'])

# Fit the model
non_norm_history = model_11.fit(train_data,
                                train_labels,
                                epochs=10,
                                validation_data=(test_data, test_labels))

# Check summary
print(model_11.summary())

# Check min and max values of training data
print(train_data.min())
print(train_data.max())


# Normalize the data
# Divide train and test imagaed by maximum values
train_data = train_data/255.0
test_data = test_data/255.0

# Check the min and max values now

print(train_data.min())
print(train_data.max())

# Build model
# Sset seed
tf.random.set_seed(42)

# create the model
model_12 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model_12.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                 optimizer=tf.keras.optimizers.Adam(),
                 metrics=['accuracy'])

# Fit the model
norm_history = model_12.fit(train_data,
                                train_labels,
                                epochs=10,
                                validation_data=(test_data, test_labels))

# Check summary
print(model_12.summary())

import pandas as pd
# plot non normalized data loss values
pd.DataFrame(non_norm_history.history).plot(title="Non normalized data")

# Plot normalized data loss curve
pd.DataFrame(norm_history.history).plot(title="Normalized data")
plt.show()

# Set random seed
tf.random.set_seed(42)

# create model

model_13 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
    ])

# compile the model
model_13.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                 optimizer=tf.keras.optimizers.Adam(),
                 metrics=['accuracy'])

# create the learning rate call back

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 10**(epoch/20))

# fit the model
find_lr_history = model_13.fit(train_data,
                               train_labels,
                               epochs=40,
                               validation_data=(test_data, test_labels),
                               callbacks=[lr_scheduler])

# plot the learning rate decay curve
import numpy as np
import matplotlib.pyplot as plt
lrs = 1e-3 * (10**(np.arange(40)/20))
plt.semilogx(lrs, find_lr_history.history['loss'])
plt.xlabel("Learning rate")
plt.ylabel("Loss")
plt.title("Finding the ideal learning rate")
plt.show()

# model using ideal learning rate

# set random seed
tf.random.set_seed(42)

# create the model
model_14 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the mdoel

model_14.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                 optimizer=tf.keras.optimizers.Adam(lr=0.001),
                 metrics=['accuracy'])
# fit the model
history = model_14.fit(train_data,
                      train_labels,
                      epochs=20,
                      validation_data=(test_data, test_labels))

from tensorflow.keras.utils import plot_model
plot_model(model_14, show_shapes=True)
