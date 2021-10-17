import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
print(tf.__version__)
# create features
x = np.array([-7.0, -4.0, -1.0, 2.0, 5.0, 8.0, 11.0, 14.0])

# Create labels
y = np.array([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0])

# Visualize it

plt.scatter(x, y)
plt.show()

# take a single example of x
input_shape = x[0].shape
# take a single example of y
output_shape = y[0].shape

print(input_shape), print(output_shape)

# set random seed

tf.random.set_seed(42)

# create a model using the sequential API
model = tf.keras.Sequential([tf.keras.layers.Dense(1)])

# compile the model
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=['mae'])

# fit model
model.fit(x, y, epochs=100)


print(model.predict([17.0]))

print(model.summary())

# make a bigger dataset

x = np.arange(-100, 100, 4)
print(x)
y = np.arange(-90, 110, 4)
print(y)
print(len(x))

x_train = x[:40]
y_train = y[:40]

x_test = x[40:]
y_test = y[40:]

print(len(x_train))
print(len(x_test))

plt.figure(figsize=(10, 7))
# Plot training data in blue
plt.scatter(x_train, y_train, c='b', label='Training Data')

# Plot test data in green
plt.scatter(x_test, y_test, c='g', label='Testing Data')

# SHow the legend
plt.legend()
plt.show()

# Build model
# set random seed

tf.random.set_seed(42)

# Create a model
model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=[1])])

# Compile model
model.compile(loss='mae',
              optimizer='SGD',
              metrics=['mae'])

# doesnt work(model not fin/built)

print(model.summary())

y_predict = model.predict(x_test)
print(y_predict)


# Plotting the predictions

def plot_predictions(train_data=x_train,
                     train_labels=y_train,
                     test_data=x_test,
                     test_labels=y_test,
                     predictions=y_predict):
    plt.figure(figsize=(10, 7))
    # Plot training data in blue
    plt.scatter(train_data, train_labels, c='b', label="Training Data")
    # Plot test data in green
    plt.scatter(test_data, test_labels, c='g', label='Testing Data')
    # Plot predictions in red
    plt.scatter(test_data, predictions, c='r', label='Predictions')

    # show legend,
    plt.legend()
    plt.show()

plot_predictions(train_data=x_train,
                 train_labels=y_train,
                 test_data=x_test,
                 test_labels=y_test,
                 predictions=y_predict)

print(model.evaluate(x_test, y_test))

# Calculate the mean absolute error

mae = tf.metrics.mean_absolute_error(y_true= y_test,
                                     y_pred= y_predict)
print(mae)

print(y_test)

print(y_predict)

print(y_test.shape)
print(y_predict.shape)

print(y_predict.squeeze().shape)


print(y_test)
print(y_predict.squeeze())

mae = tf.metrics.mean_absolute_error(y_true = y_test,
                                     y_pred= y_predict.squeeze())
print(mae)

print(tf.reduce_mean(tf.abs(y_test-y_predict.squeeze())))

# Make function for prediction summary

def mae(y_test, y_pred):
    #Calclulate mean absolute error between y_test and y_preds
    return tf.metrics.mean_absolute_error(y_test,
                                          y_pred)

def  mse(ytest, y_pred):
    # Calculate mean absolute error
    return tf.metrics.mean_squared_error(y_test,
                                         y_pred)

# Build model_1

    # set random seed
tf.random.set_seed(42)

#Replicate original model
model_1 = tf.keras.Sequential([
    tf.keras.layers.Dense(1)
])

# Compile the model

model_1.compile(loss= tf.keras.losses.mae,
                optimizer= tf.keras.optimizers.SGD(),
                metrics= ['mae'])

# Fit the model

model_1.fit(x_train, y_train, epochs=100)

#  Make and plot predictions for model_1

y_preds_1 = model_1.predict(x_test)
plot_predictions(predictions=y_preds_1)

mae_1 = mae(y_test, y_preds_1.squeeze()).numpy()
mse_1 = mse(y_test, y_preds_1.squeeze()).numpy()
print(mae_1)
print(mse_1)

# Build model_2

# set random seed

tf.random.set_seed(42)

# Replicate model_1 and add an extra layer

model_2 = tf.keras.Sequential([
    tf.keras.layers.Dense(1),
    tf.keras.layers.Dense(1)
])

# Compile the model
model_2.compile(loss = tf.keras.losses.mae,
                optimizer= tf.keras.optimizers.SGD(),
                metrics=['mae'])

# Fit the model

model_2.fit(x_train, y_train, epochs=100, verbose=0)

# Make and plot predictions for model_2

y_preds_2 = model_2.predict(x_test)
plot_predictions(predictions=y_preds_2)

# Calculate model_2 metrics

mae_2 = mae(y_test, y_preds_2.squeeze()).numpy()
mse_2 = mse(y_test, y_preds_2.squeeze()).numpy()

print(mae_2)
print(mse_2)

# Build model_3

# set random seed

tf.random.set_seed(42)

# Replicate model_2
model_3 = tf.keras.Sequential([
    tf.keras.layers.Dense(1),
    tf.keras.layers.Dense(1)
])

# Compile the model
model_3.compile(loss=tf.keras.losses.mae,
                optimizer=tf.keras.optimizers.SGD(),
                metrics=['mae'])

#Fit the model

model_3.fit(x_train, y_train, epochs=500, verbose=0)

#make and plot predictions

y_preds_3 = model_3.predict(x_test)
plot_predictions(predictions=y_preds_3)

# Calculate model_3 metrics

mae_3 = mae(y_test, y_preds_3.squeeze()).numpy()
mse_3 = mse(y_test, y_preds_3.squeeze()).numpy()
print(mae_3)
print(mse_3)


model_results = [["model_1", mae_1, mse_1],
                 ["model_2", mae_2, mse_2],
                 ["model_3", mae_3, mse_3]]

import pandas as pd
#all_results = pd.DataFrame(model_resluts, columns=['model', 'mae', 'mse'])

model_2.save('best_model')

loaded_saved_model = tf.keras.models.load_model('best_model')
print(loaded_saved_model.summary())

model_2.save("best_model_hdf5.h5")
loaded_h5_model = tf.keras.models.load_model("best_model_hdf5.h5")
print(loaded_h5_model.summary())




