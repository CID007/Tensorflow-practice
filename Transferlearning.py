import zipfile
import wget

# download data
wget.download("https://storage.googleapis.com/ztm_tf_course/food_vision/10_food_classes_10_percent.zip")

#unzip the downloaded file
zip_ref = zipfile.ZipFile("10_food_classes_10_percent.zip", 'r')
zip_ref.extractall()
zip_ref.close()

# Check how many images in each folde
import os
for dirpath, dirnames, filenames in os.walk("10_food_classes_10_percent"):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

# Creating data loaders(preparing the data)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_shape = (224, 224)
batch_size = 32

train_dir = "10_food_classes_10_percent/train/"
test_dir = "10_food_classes_10_percent/test/"

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

print("training images")
train_data_10_percent = train_datagen.flow_from_directory(train_dir,
                                                          target_size=image_shape,
                                                          batch_size=batch_size,
                                                          class_mode="categorical")
print("testing images")
test_data = test_datagen.flow_from_directory(test_dir,
                                             target_size=image_shape,
                                             batch_size=batch_size,
                                             class_mode='categorical')

# Create a tensorboard call back function
import datetime


def create_tensorboard_callback(dir_name, experiment_name):
    log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir
    )
    print(f"Saving TensorBoard log files to :{log_dir}")
    return tensorboard_callback


import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers

# Resnet 50
resnet_url = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4"

# original efficientnet feature vector
efficientnet_url = "https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1"


def create_model(model_url, num_classes=10):
    feature_extractor_layer = hub.KerasLayer(model_url,
                                             trainable=False,
                                             name='feature_extraction_layer',
                                             input_shape=image_shape+(3,))
    #create our model
    model = tf.keras.Sequential([
        feature_extractor_layer,
        layers.Dense(num_classes, activation='softmax', name='output_layer')
    ])
    return model


#create model
resnet_model = create_model(resnet_url, num_classes=train_data_10_percent.num_classes)

# Compile the model
resnet_model.compile(loss='categorical_crossentropy',
                     optimizer=tf.keras.optimizers.Adam(),
                     metrics=['accuracy'])

# fit the model
resnet_history = resnet_model.fit(train_data_10_percent,
                                  epochs=5,
                                  steps_per_epoch=len(train_data_10_percent),
                                  validation_data=test_data,
                                  validation_steps=len(test_data),
                                  callbacks=[create_tensorboard_callback(dir_name="tensorflow_hub",
                                                                         experiment_name="resnet50v2")])
# Define plotting function
import  matplotlib.pyplot as plt


def plot_loss_curves(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    epochs = range(len(history.history['loss']))

    #plot loss
    plt.plot(epochs, loss, label='training loss')
    plt.plot(epochs, val_loss, label='validation loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    #plot accuracy
    plt.figure()
    plt.plot(epochs, accuracy, label='training accuracy')
    plt.plot(epochs, val_accuracy, label='validation accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()


plot_loss_curves(resnet_history)

# Create efficient net model
efficientnet_model = create_model(model_url=efficientnet_url,
                                  num_classes=train_data_10_percent.num_classes)

# Compile efficient net model
efficientnet_model.compile(loss='categorical_crossentropy',
                           optimizer=tf.keras.optimizers.Adam(),
                           metrics=['accuracy'])

# Fit efficientnet model
efficientnet_history = efficientnet_model.fit(train_data_10_percent,
                                              epochs=5,
                                              steps_per_epoch=len(train_data_10_percent),
                                              validation_data=test_data,
                                              validation_steps=len(test_data),
                                              callbacks=[create_tensorboard_callback(dir_name='tensorflow_hub',
                                                                                     experiment_name='efficientnetB0')])
plot_loss_curves(efficientnet_history)
