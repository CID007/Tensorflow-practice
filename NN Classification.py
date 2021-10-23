import keras.losses
import tensorflow as tf
print(tf.__version__)

from sklearn.datasets import make_circles

# make 1000 examples
n_samples = 1000

# Create circles

x, y = make_circles(n_samples,
                    noise=0.03,
                    random_state=42)

print(x)
print(y[:10])

# Make dataframe of features and labels
import pandas as pd
circles = pd.DataFrame({"x0":x[:, 0], "x1":x[:, 1], "label":y})
print(circles.head())

# Check out the different labels
print(circles.label.value_counts())

import matplotlib.pyplot as plt
import matplotlib.cm
plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.RdYlBu)
plt.show()

# Check the shapes of our features and labels
print(x.shape)
print(y.shape)

# set random seed
tf.random.set_seed(42)

# Create the model using Sequential
model_1 = tf.keras.Sequential([
    tf.keras.layers.Dense(1)
])

# Compile model
model_1.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=tf.keras.optimizers.SGD(),
                metrics=['accuracy'])

# Fit the model
model_1.fit(x, y, epochs=5)

# Train our model for longer
model_1.fit(x, y, epochs=200, verbose=0)
model_1.evaluate(x, y)

# Add extra layer and train for longer

# Set random seed
tf.random.set_seed(42)

# Create model 2
model_2 = tf.keras.Sequential([
    tf.keras.layers.Dense(1),
    tf.keras.layers.Dense(1)
])

# Compile the model
model_2.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=tf.keras.optimizers.SGD(),
                metrics=['accuracy'])

# Fit the model
model_2.fit(x, y, epochs=100, verbose=0)

# Evaluate model 2
print(model_2.evaluate(x, y))

# Model 3 with more hyper parameters
# set random seed
tf.random.set_seed(42)

# Create model
model_3 = tf.keras.Sequential([
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
])

# Compile the model,

model_3.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"]
                )

# Fit the model
print(model_3.fit(x, y, epochs=100, verbose=0))
import numpy as np
def plot_decision_boundary(model, x, y):
    # Define the axis boundaries of the plot and create a mesh grid
    x_min, x_max = x[:, 0].min() - 0.1, x[:, 0].max() + 0.1
    y_min, y_max = x[:, 1].min() - 0.1, x[:, 1].max() + 0.1

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    # create x values
    x_in = np.c_[xx.ravel(), yy.ravel()]

    # make predictions using the trained model
    y_pred = model.predict(x_in)

    # Check for multi-class
    if len(y_pred[0]) > 1:
        print("doing multiclass classification...")
        y_pred = np.argmax(y_pred, axis=1).reshape(xx.shape)
    else:
        print("doing binary classification...")
        y_pred = np.round(y_pred).reshape(xx.shape)

    # Plot decision boundary
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(x[:, 0], x[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()

# check out the precictions our model is making
plot_decision_boundary(model_3, x, y)

#Check using regression model.

# Set random seed
tf.random.set_seed(42)
x_regression = np.arange(0, 1000, 5)
y_regression = np.arange(100, 1100, 5)

# split into training and validation sets
x_reg_train = x_regression[:150]
x_reg_test = x_regression[150:]
y_reg_train = y_regression[:150]
y_reg_test = y_regression[150:]
# Recreate the model
model_3 = tf.keras.Sequential([
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
])

# Change the lsos and metrics of our compiled model
model_3.compile(loss=tf.keras.losses.mae,
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['mae'])

# Fit the model
model_3.fit(x_reg_train, y_reg_train, epochs=100)

# Make predictions with our trained model
y_reg_preds = model_3.predict(y_reg_test)

# Plot the model's predictions
plt.figure(figsize=(10, 7))
plt.scatter(x_reg_train, y_reg_train, c='b', label='Training data')
plt.scatter(x_reg_test, y_reg_test, c='g', label='Testing data')
plt.scatter(x_reg_test, y_reg_preds.squeeze(), c='r', label="Predictions")
plt.legend()
plt.show()

# Add non linearity in model

# set random seed
tf.random.set_seed(42)

# Create the model
model_4 = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation=tf.keras.activations.linear),
    tf.keras.layers.Dense(1)
])

# Compile the model
model_4.compile(loss=tf.keras.losses.binary_crossentropy,
                optimizer=tf.keras.optimizers.Adam(lr=0.001),
                metrics=['accuracy'])

# fit the model
history = model_4.fit(x, y, epochs=100)

# Check out our data
plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.RdYlBu)
plt.show()

# Check the decision boundary (blue id blue class , yellow is

plot_decision_boundary(model_4, x, y)
plt.show()

# Add non linearity
# Set random seed
tf.random.set_seed(42)

# Create model with a non linear activation
model_5 = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(1)
])

# Compile the model
model_5.compile(loss=tf.keras.losses.binary_crossentropy,
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['accuracy'])

# Fit the model

model_5.fit(x, y, epochs=100)

# Check decision boundary

plot_decision_boundary(model_5, x, y)
plt.show()


# Define model 6 with more hidden layers and more neurons

# set random seed
tf.random.set_seed(42)

# Create model
model_6 = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(4, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(1)
])

# compile the model

model_6.compile(loss=tf.keras.losses.binary_crossentropy,
                optimizer=tf.keras.optimizers.Adam(lr=0.001),
                metrics=["accuracy"])

# fit the model
history = model_6.fit(x, y, epochs=100)

# Evaluate the model

model_6.evaluate(x, y)

# plot decision boundary
plot_decision_boundary(model_6, x, y)


#change activation function for output layer too

# set random seed
tf.random.set_seed(42)

#create a model
model_7 = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(4, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)
])

# Compile the model
model_7.compile(loss=tf.keras.losses.binary_crossentropy,
                optimizer=tf.keras.optimizers.Adam(lr=0.001),
                metrics=["accuracy"])

#Evaluate the model
model_7.evaluate(x, y)

# Plot decision boundary

plot_decision_boundary(model_7, x, y)

####################################################
# Create a toy tensor
a = tf.cast(tf.range(-10, 10), tf.float32)
print(a)
plt.plot(a)
plt.show()


# Sigmoid function
def sigmoid(x):
    return 1 / (1 + tf.exp(-x))

sigmoid(a)
plt.plot(sigmoid(a))
plt.show()

# Def relu

def relu(x):
    return tf.maximum(0, x)

relu(a)
plt.plot(relu(a))
plt.show()

# split dta into train and test sets
x_train, y_train = x[:800], y[:800]
x_test, y_test = x[800:], y[800:]

#check the shape of the data
print(x_train.shape)
print(x_test.shape)

# Define new model
# set random seed
tf.random.set_seed(42)

# Create the model
model_8 = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model_8.compile(loss=tf.keras.losses.binary_crossentropy,
                optimizer=tf.keras.optimizers.Adam(lr=0.01),
                metrics=["accuracy"])

# fit the model
hisgory = model_8.fit(x_train, y_train, epochs=25)
loss, accuracy = model_8.evaluate(x_test, y_test)
print(f"Model loss on the test set: {loss}")
print(f"Model accuracy on the test set: {100*accuracy: .2f}%")
#plot decision boundary for trainin and test sets
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_8, x=x_train, y=y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_8, x=x_test, y=y_test)
plt.show()

#plot loss curve
print(pd.DataFrame(history.history))

# Plot the loss curves
pd.DataFrame(history.history).plot()
plt.title("MOdel_8 trianing curves")
plt.show()

# Another model with call back

#set random seed
tf.random.set_seed(42)

# create model
model_9 = tf.keras.models.Sequential([
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# compile the modell
model_9.compile(loss='binary_crossentropy',
                optimizer="Adam",
                metrics=["accuracy"])

# create a learning rate scheduler call back
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10**(epoch/20))

#fit the model
history = model_9.fit(x_train,
                      y_train,
                      epochs=100,
                      callbacks=[lr_scheduler])

# checkout the history
pd.DataFrame(history.history).plot(figsize=(10, 7), xlabel="epoch")

# plot learning rate vs the loss
lrs = 1e-4 * (10**(np.arange(100)/20))
plt.figure(figsize=(10, 7))
plt.semilogx(lrs, history.history["loss"])
plt.xlabel("Learning Rate")
plt.ylabel("Loss")
plt.title("Learning rate vs loss")
plt.show()

# define new model with lr =0.2

# set random seed
tf.random.set_seed(42)

# Create the model
model_10 = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model_10.compile(loss="binary_crossentropy",
                 optimizer=tf.keras.optimizers.Adam(lr=0.02),
                 metrics=["accuracy"])

# Fit the model
history = model_10.fit(x_train, y_train, epochs=20)

# Evaluate the model
model_10.evaluate(x_test, y_test)

# Plot the results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2 , 1)
plt.title("Train")
plot_decision_boundary(model_10, x=x_train, y=y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_10, x=x_test, y=y_test)
plt.show()


# Check the accuracy of our model
loss, accuracy = model_10.evaluate(x_test, y_test)
print(f"Model loss on test set : {loss}")
print(f"Model accuracy on test set: {(accuracy*100):.2f}%")

# Create confusion matric
from sklearn.metrics import confusion_matrix

# make predictions
y_preds = model_10.predict(x_test)

# create confusion matrix
print(confusion_matrix(y_test, tf.round(y_preds)))


