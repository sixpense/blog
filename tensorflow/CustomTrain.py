import tensorflow as tf
tf.enable_eager_execution()

class Model(object):
    def __init__(self):
        # Initialize variable to (5.0, 0.0)
        # inn practice, these should be initailized to random values.
        self.W = tf.Variable(5.0)
        self.b = tf.Variable(0.0)

    def __call__(self, x):
        return self.W * x +self.b

model = Model()

assert model(3.0).numpy() == 15.0

def loss(predicted_y, desired_y):
    return tf.reduce_mean(tf.square(predicted_y - desired_y))

TRUE_W = 3.0
TRUE_b = 2.0
NUM_EXAMPLES = 1000

inputs = tf.random_normal(shape = [NUM_EXAMPLES])
noise = tf.random_normal(shape = [NUM_EXAMPLES])
outputs = inputs * TRUE_W + TRUE_b + noise

import matplotlib.pyplot as plt

plt.scatter(inputs, outputs, c = 'b')
plt.scatter(inputs, model(inputs), c = 'r')
plt.show()

print('Current loss:'),
print(loss(model(inputs), outputs).numpy())
