import numpy as np
import tensorflow as tf
import itertools
import random


# Create dataset
dataset = np.arange(999, dtype=np.float32) + 1

# Get all of the permutations as pairs: [ (2,3,6), (7,2,14), (5,5,25), (1,9,9), (110, 5, 550), ...]
pairs = [(a / 999.0, b / 999.0, (a * b) / (999.0 * 999.0)) for (a, b) in list(itertools.permutations(dataset, 2))]
pairs = [(a, b, (a * b)) for (a, b) in list(itertools.combinations_with_replacement(dataset, 2))]
n_pairs = len(pairs)

# Model's Parameters
n_input = 2  # InputLayer dimensions
n_hidden_1 = 20  # HiddenLayer 1 dimensions
n_hidden_2 = 20  # HiddenLayer 2 dimensions
n_output = 1  # OutputLayer dimensions

# Tensorflow Graph Input
X = tf.placeholder("float32", [None, n_input])
Y = tf.placeholder("float32", [None, n_output])


# Create MLP Model
def create_MLP(x, weights, biases):
    hidden_layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    hidden_layer_1 = tf.nn.relu(hidden_layer_1)

    hidden_layer_2 = tf.add(tf.matmul(hidden_layer_1, weights['h2']), biases['b2'])
    hidden_layer_2 = tf.nn.relu(hidden_layer_2)

    output_layer = tf.matmul(hidden_layer_2, weights['out']) + biases['out']
    output_layer = tf.nn.sigmoid(output_layer)

    return output_layer


# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_output]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_output]))
}

# Construct model
output_layer = create_MLP(X, weights, biases)

# Model's HyperParameters
epochs = 10000000
batch_size = 100

# Define Loss Functions
mse_loss = tf.reduce_sum(tf.pow(output_layer - Y, 2)) / (2 * batch_size)

# Define Optimizer (SGD, Adam, RMSProp, Adagrad, AdaDelta, ...)
learning_rate = 0.001
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(mse_loss)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(mse_loss)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the Tensorflow Graph
with tf.Session() as sess:
    sess.run(init)

    # Do epochs
    for epoch in range(epochs):
        batch = np.array(random.sample(pairs, batch_size))
        X_train = batch[:, :2]
        Y_train = batch[:, 2:]
        optimizer_value, mse_loss_value, output_layer_value = sess.run([optimizer, mse_loss, output_layer], feed_dict={X: X_train, Y: Y_train})

        # Display Results Every 1000 epochs
        if epoch % 1000 == 0:
            print("Normalized Results:")
            for index, item in enumerate(batch):
                print('%s * %s = %s || %s' % (item[0], item[1], item[2], output_layer_value[index]))
                if index >= 0:
                    break
            print('Loss => %s\n' % mse_loss_value)

            print("DeNormalized Results:")
            test_batch = np.array(random.sample(pairs, 1))
            test_prediction_values = sess.run([output_layer], feed_dict={X: test_batch[:, :2]})
            for j, test_item in enumerate(test_batch):
                print('%s * %s = %s || %s\n' % (int(test_item[0] * 999), int(test_item[1] * 999), int(test_item[2] * (999 * 999)), int(test_prediction_values[0][j] * (999 * 999))))
