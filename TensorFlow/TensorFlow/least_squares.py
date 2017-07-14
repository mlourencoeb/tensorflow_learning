# required imports
import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_california_housing

# read and debug data dimenions
housing = fetch_california_housing()
m,n  = housing.data.shape
print(m)
print(n)

print(housing.data.shape)

#concatenate housing price with ones at the begining.
housing_data_and_bias = np.c_[np.ones((m,1)), housing.data]

# declare variables for performing least squares with normal equations
X = tf.constant(housing_data_and_bias, dtype=tf.float32, name="X")
Y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
XT = tf.transpose(X)

# actual computations ot theta
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT,X)), XT), Y)
print(theta)

with tf.Session() as sess:
    theta_value = theta.eval()