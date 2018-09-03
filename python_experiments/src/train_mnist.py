"""Maxout Activation Units Experiments.

    Comparison of ReLU, Maxout and lifting activations on the example of MNIST
    image classification.

    Code inspired by https://github.com/philipperemy/tensorflow-maxout.
"""
from __future__ import print_function

import sys
import numpy as np

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def create_weight_variable(name, shape):
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    return tf.Variable(initializer(shape=shape), name=name)


def create_bias_variable(name, shape):
    initializer = tf.constant_initializer(value=0.0, dtype=tf.float32)
    return tf.Variable(initializer(shape=shape), name=name)

np.random.seed(1)
tf.set_random_seed(1)

tf.logging.set_verbosity(tf.logging.ERROR)
mnist = input_data.read_data_sets('data', one_hot=True)

# Parameters
learning_rate = 0.01
training_epochs = 100
batch_size = 100
display_step = 1

# tf Graph Input
# mnist data image of shape 28*28=784
x = tf.placeholder(tf.float32, [None, 784], name='InputData')
# 0-9 digits recognition => 10 classes
y = tf.placeholder(tf.float32, [None, 10], name='LabelData')


def linear(): W1 = create_weight_variable('Weights', [784, 10]):
    b1 = create_bias_variable('Bias', [10])
    return tf.nn.softmax(tf.matmul(x, W1) + b1)


def hidden_relu():
    W1 = create_weight_variable('Weights', [784, 100])
    b1 = create_bias_variable('Bias', [100])

    W2 = create_weight_variable('Weights2', [100, 10])
    b2 = create_bias_variable('Bias2', [10])
    t = tf.nn.relu(tf.matmul(x, W1) + b1)
    return tf.nn.softmax(tf.matmul(t, W2) + b2)


def max_out(inputs, num_units, axis=None):
    shape = inputs.get_shape().as_list()
    if shape[0] is None:
        shape[0] = -1
    if axis is None:  # Assume that channel is the last dimension
        axis = -1
    num_channels = shape[axis]
    if num_channels % num_units:
        raise ValueError('number of features({}) is not '
                         'a multiple of num_units({})'.format(num_channels, num_units))
    shape[axis] = num_units
    shape += [num_channels // num_units]
    outputs = tf.reduce_max(tf.reshape(inputs, shape), -1, keep_dims=False)
    return outputs


def hidden_maxout():
    W1 = create_weight_variable('Weights', [784, 100])
    b1 = create_bias_variable('Bias', [100])

    W2 = create_weight_variable('Weights2', [50, 10])
    b2 = create_bias_variable('Bias2', [10])

    t = max_out(tf.matmul(x, W1) + b1, 50)
    return tf.nn.softmax(tf.matmul(t, W2) + b2)


def lift_layer(input):
    output = tf.concat([tf.nn.relu(input), -1.0 * tf.nn.relu(-1.0 * input)], axis=1)
    return output


def hidden_lift():
    W1 = create_weight_variable('Weights', [784, 95])
    b1 = create_bias_variable('Bias', [95])

    W2 = create_weight_variable('Weights2', [190, 10])
    b2 = create_bias_variable('Bias2', [10])

    t = lift_layer(tf.matmul(x, W1) + b1)
    return tf.nn.softmax(tf.matmul(t, W2) + b2)


def select_model():
    usage = 'Usage: python mnist_maxout_example.py (LINEAR|RELU|MAXOUT|LIFT)'
    assert len(sys.argv) == 2, usage
    t = sys.argv[1].upper()
    print('Type = ' + t)
    if t == 'LINEAR':
        return linear(), 'logs/linear'
    elif t == 'RELU':
        return hidden_relu(), 'logs/relu'
    elif t == 'MAXOUT':
        return hidden_maxout(), 'logs/maxout'
    elif t == 'LIFT':
        return hidden_lift(), 'logs/lift'
    else:
        raise Exception('Unknown type. ' + usage)


# Construct model and encapsulating all ops into scopes, making
# Tensorboard's Graph visualization more convenient
with tf.name_scope('Model'):
    # Model
    pred, logs_path = select_model()
with tf.name_scope('Loss'):
    # Minimize error using cross entropy
    cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))
with tf.name_scope('SGD'):
    # Gradient Descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
with tf.name_scope('Accuracy'):
    # Accuracy
    acc = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    acc = tf.reduce_mean(tf.cast(acc, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()
# Merge all summaries into a single op
train_summary = tf.summary.merge([tf.summary.scalar('train_loss', cost),
                                  tf.summary.scalar('train_accuracy', acc)])
test_summary = tf.summary.merge([tf.summary.scalar('test_loss', cost),
                                 tf.summary.scalar('test_accuracy', acc)])

# Launch the graph
sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
with tf.Session(config=sess_config) as sess:
    sess.run(init)
    print(f"NUM TRAIN PARAMS: {np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])}")

    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop), cost op (to get loss value)
            # and summary nodes
            _, c, summary = sess.run([optimizer, cost, train_summary],
                                     feed_dict={x: batch_xs, y: batch_ys})
            # Write logs at every iteration
            summary_writer.add_summary(summary, epoch * total_batch + i)
            # Compute average loss
            avg_cost += c / total_batch
        summary = test_summary.eval({x: mnist.test.images, y: mnist.test.labels})
        summary_writer.add_summary(summary, epoch + 1)
        # Display logs per epoch step
        # if (epoch + 1) % display_step == 0:
        #     print('Epoch:', '%04d' % (epoch + 1), 'cost=', '{:.9f}'.format(avg_cost))

    print('Optimization Finished!')

    # Test model
    # Calculate accuracy
    print('Accuracy: ', acc.eval({x: mnist.test.images, y: mnist.test.labels}))
    print(f"Loss: {avg_cost}")

    print('Run the command line:\n' \
          '--> tensorboard --logdir=/tmp/tensorflow_logs ' \
          '\nThen open http://0.0.0.0:6006/ into your web browser')
