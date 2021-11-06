import wget
wget.download("https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/extras/helper_functions.py")

#Import helper functions we are going to use

from helper_functions import create_tensorboard_callback, plot_loss_curves, walk_through_dir, unzip_data

#get 10% of the data of the 10 classes
wget.download("https://storage.googleapis.com/ztm_tf_course/food_vision/10_food_classes_10_percent.zip")
unzip_data("10_food_classes_10_percent.zip")
walk_through_dir("10_food_classes_10_percent")
#create train and test directories

train_dir = "10_food_classes_10_percent/train/"
test_dir = "10_food_classes_10_percent/test"

# create data inputs
import tensorflow as tf
img_size = (224, 224)
train_data_10_percent = tf.keras.preprocessing.image_dataset_from_directory(directory=train_dir,
                                                                            image_size=img_size,
                                                                            label_mode="categorical",
                                                                            batch_size=32)
test_data_10_percent = tf.keras.preprocessing.image_dataset_from_directory(directory=test_dir,
                                                                           image_size=img_size,
                                                                           label_mode="categorical"
                                                                           )
# check the training data data tpe
print(train_data_10_percent)
print(train_data_10_percent.class_names)

# see the examples of batch of data
for images, labels in train_data_10_percent.take(1):
    print(images, labels)

#1 Create base model
base_model = tf.keras.applications.EfficientNetB0(include_top=False)

#2 Freeze the base model
base_model.trainable = False

#3. Creat imputs to the base model
inputs = tf.keras.layers.Input(shape=(224, 224, 3), name="input_layer")

#5 Pass input the th base model
x = base_model(inputs)
#check shape after passing it to base_model
print(f"shape after base_model: {x.shape}")
#6. Avg pool output of base model
x = tf.keras.layers.GlobalAveragePooling2D(name="Global_average_pooling_layer")(x)

#7 Create the output activation layer
outputs = tf.keras.layers.Dense(10, activation='softmax', name="Output_layer")(x)

# Combine inputs with outputs
model_0 = tf.keras.Model(inputs, outputs)

model_0.compile(loss="categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['accuracy'])

#10. Fit the model
history_10_percent = model_0.fit(train_data_10_percent,
                                 epochs=5,
                                 steps_per_epoch=len(train_data_10_percent),
                                 validation_data=test_data_10_percent,
                                 validation_steps=int(0.25*len(test_data_10_percent)),
                                 callbacks=[create_tensorboard_callback("transfer_learning", "10_percent_feature_extract")])

#Check layers in our base model
for layer_number, layer in enumerate(base_model.layers):
    print(layer_number, layer.name)

#print model summary
print(base_model.summary())

#Check summary of model constructed with functional API
print(model_0.summary())

# Check our model's training curves
plot_loss_curves(history_10_percent)

#Download and unzip 1 % data
wget.download("https://storage.googleapis.com/ztm_tf_course/food_vision/10_food_classes_1_percent.zip")
unzip_data("10_food_classes_1_percent.zip")

#create training and test dirs
train_dir_1_percent = "10_food_classes_1_percent/train/"
test_dir = "10_food_classes_1_percent/test/"

# Walk through 1 percent data directory and list number of files
walk_through_dir("10_food_classes_1_percent")

import tensorflow as tf
img_size = (224, 224)
train_data_1_percent = tf.keras.preprocessing.image_dataset_from_directory(train_dir_1_percent,
                                                                           label_mode="categorical",
                                                                           batch_size=32,
                                                                           image_size=img_size)
test_data = tf.keras.preprocessing.image_dataset_from_directory(test_dir,
                                                                label_mode="categorical",
                                                                image_size=img_size)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

# create data augmentation stage
data_augmentation = keras.Sequential([
    preprocessing.RandomFlip("horizontal"),
    preprocessing.RandomRotation(0.2),
    preprocessing.RandomZoom(0.2),
    preprocessing.RandomHeight(0.2),
    preprocessing.RandomWidth(0.2),

], name='data_augmentation')

# View random image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import random
target_class = random.choice(train_data_1_percent.class_names)
target_dir = "10_food_classes_1_percent/train/" + target_class
random_image = random.choice(os.listdir(target_dir))
random_image_path = target_dir + "/" + random_image
img = mpimg.imread(random_image_path)
plt.imshow(img)
plt.title(f"Original random image from class : {target_class}")
plt.axis(False)
plt.show()

# Augment the image
augmented_img = data_augmentation(tf.expand_dims(img, axis=0))
plt.figure()
plt.imshow(tf.squeeze(augmented_img)/255.)
plt.title(f"Augmented random image from class: {target_class}")
plt.axis(False)

#Model 1 : Feature extration transfer learning on 1 % data
# Set up input shape and base model

input_shape = (224, 224, 3)
base_model = tf.keras.applications.EfficientNetB0(include_top=False)
base_model.trainable = False

# Creat input layer
inputs = layers.Input(shape=input_shape, name='input_layer')

# Add in data augmentation layer
x = data_augmentation(inputs)

# Give base model inputs
x = base_model(x, training=False)

# Pool output feature of base model
x = layers.GlobalAveragePooling2D(name='Global_average_pooling_layer')(x)
# PUt a dense layer on the output
outputs = layers.Dense(10, activation='softmax', name='output_layer')(x)
model_1 = keras.Model(inputs, outputs)
#Compile the model
model_1.compile(loss="categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['accuracy'])

# Fit the model
history_1_percent = model_1.fit(train_dir_1_percent,
                                epochs=5,
                                steps_per_epoch=len(train_dir_1_percent),
                                validation_data=test_data,
                                validation_steps=len(test_data),
                                callbacks=[create_tensorboard_callback("transfer_learning", "1_percent_data_aug")])
print(model_1.summary())

# Evaluate on the test data
results_1_percent_data_aug = model_1.evaluate(test_data)
print(results_1_percent_data_aug)

plot_loss_curves(history_1_percent)
plt.show()
