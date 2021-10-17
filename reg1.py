# import required libraries
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

# Read in the insurance dataset

insurance = pd.read_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv")
print(insurance.head())

# Turn all categories into numbers

insurance_one_hot = pd.get_dummies(insurance)
print(insurance_one_hot.head())

# Create x and y values

x = insurance_one_hot.drop("charges", axis=1)
y = insurance_one_hot["charges"]

print(x.head())

# Create training and test sets

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Set random seed
tf.random.set_seed(42)

# Create a new model
insurance_model = tf.keras.Sequential([
    tf.keras.layers.Dense(1),
    tf.keras.layers.Dense(1)
])

# Compile the model
insurance_model.compile(loss=tf.keras.losses.mae,
                        optimizer=tf.keras.optimizers.SGD(),
                        metrics=['mae'])

# Fit the model
insurance_model.fit(x_train, y_train, epochs=100)

# CHeck results

print(insurance_model.evaluate(x_test, y_test))

##########################################

# lets try a bigger modl
# Set random seed
tf.random.set_seed(42)

# Add an extra layer and increase number of units
insurance_model_2 = tf.keras.Sequential([
  tf.keras.layers.Dense(100), # 100 units
  tf.keras.layers.Dense(10), # 10 units
  tf.keras.layers.Dense(1) # 1 unit (important for output layer)
])

# Compile model
insurance_model_2.compile(loss=tf.keras.losses.mae,
                          optimizer=tf.keras.optimizers.Adam(),
                          metrics=['mae'])

# Fit the model
history = insurance_model_2.fit(x_train, y_train, epochs=100, verbose=0)

# Evaluate our larger model

print(insurance_model_2.evaluate(x_test, y_test))
# Plot history
pd.DataFrame(history.history).plot()
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.show()

history_2 = insurance_model_2.fit(x_train, y_train, epochs=100, verbose=0)

# Evaluate the model trained for 200 total epochs
insurance_model_2_loss, insurance_model_2_mae = insurance_model_2.evaluate(x_test, y_test)
print(insurance_model_2_loss)
print(insurance_model_2_mae)

# Plot the model trained for 200 total epochs loss curve
pd.DataFrame(history_2.history).plot()
plt.ylabel("loss")
plt.ylabel("Epochs")
plt.show()

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Create column transformer

ct = make_column_transformer(
    (MinMaxScaler(), ["age", "bmi", "children"]),
    (OneHotEncoder(handle_unknown='ignore'), ["sex", "smoker", "region"])
)

# Create x and y
x = insurance.drop("charges", axis=1)
y = insurance["charges"]

# Build our train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Fit column transformer
ct.fit(x_train)

# Transform training and test data with normalization
x_train_normal = ct.transform(x_train)
x_test_normal = ct.transform(x_test)

print(x_train.loc[0])

print(x_train_normal[0])

print(x_train_normal.shape)
print(x_train.shape)


# Data is normalized now
# set random seed
tf.random.set_seed(42)

# Build the model
insurance_model_3 = tf.keras.Sequential([
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
])

# Compile the model
insurance_model_3.compile(loss=tf.keras.losses.mae,
                          optimizer=tf.keras.optimizers.Adam(),
                          metrics=['mae'])

# FIt the model
insurance_model_3.fit(x_train_normal, y_train, epochs=100, verbose=0)

# Evaluate third model
insurance_model_3_loss, insurance_model_3_mae = insurance_model_3.evaluate(x_test_normal, y_test)

# Compare modelling results from non-normalized data and normalized data
print(insurance_model_2_mae)
print(insurance_model_3_mae)