import tensorflow as tf
# import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
print(tf.__version__)

scalar = tf.constant(7)
print(scalar)

print(scalar.ndim)

vector = tf.constant([10, 10])
print(vector)
print(vector.ndim)

matrix = tf.constant([[10, 7],
                     [7, 10]])

print(matrix)
print(matrix.ndim)
another_matrix = tf.constant([[10., 7.],
                              [3., 2.],
                              [8., 9.]], dtype=tf.float16)

print(another_matrix)
print(another_matrix.ndim)

tensor = tf.constant([[[1, 2, 3],
                       [4, 5, 6]],
                     [[7, 8, 9],
                      [10, 11, 12]],
                     [[13, 14, 15],
                      [16, 17, 18]]])
print(tensor)
print(tensor.ndim)
random_1 = tf.random.Generator.from_seed(42)
random_1 = random_1.normal(shape=(3, 2))
random_2 = tf.random.Generator.from_seed(42)
random_2 = random_2.normal(shape=(3, 2))
print(random_1), print(random_2)
print(random_1 == random_2)

random_3 = tf.random.Generator.from_seed(42)
random_3 = random_3.normal(shape=(3, 2))
random_4 = tf.random.Generator.from_seed(11)
random_4 = random_4.normal(shape=(3, 2))
print(random_3),
print(random_4)
print(random_1 == random_4)
print(random_3 == random_4)

not_shuffled = tf.constant([[10, 7],
                            [3, 4],
                            [2, 5]])
print(tf.random.shuffle(not_shuffled))

# make tensor of all ones
print(tf.ones(shape=(3, 2)))
print(tf.zeros(shape=(3, 2)))

numpy_a = np.arange(1, 25, dtype=np.int32)
a = tf.constant(numpy_a, shape=[2, 4, 3])
print(numpy_a)
print(a)
tensor = tf.constant([[10, 7], [3, 4]])
print(tensor)
tensor+10

tf.multiply(tensor, 10)

tensor@tensor
print(tf.reshape(tensor, shape=(1, 4)))
print(tensor)
print(tf.argmax(tensor))

some_list = [0, 1, 2, 3]
print(tf.one_hot(some_list, depth=4))

