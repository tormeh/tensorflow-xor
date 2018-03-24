from __future__ import print_function
import numpy as np
import tensorflow as tf
sess = tf.Session()

def main():
    W2 = tf.Variable(tf.random_normal([1,2]), dtype=tf.float32)
    W = tf.Variable(tf.random_normal([2,2]), dtype=tf.float32)
    b = tf.Variable(tf.random_normal([2,1]), dtype=tf.float32)
    b2 = tf.Variable([-.3], dtype=tf.float32)
    x = tf.placeholder(tf.float32)
    x_matrix = tf.transpose(x)
    layer1 = tf.sigmoid(tf.tensordot(W,x_matrix, 1) + b)
    #layer2 = tf.tensordot(W2,tf.nn.relu(layer1), 1)+b2
    layer2 = tf.sigmoid(tf.tensordot(W2,layer1, 1)+b2)
    model = layer2
    #model = tf.tensordot(W2,x_matrix, 1) + b2
    #
    init = tf.global_variables_initializer()
    sess.run(init)
    #
    y = tf.placeholder(tf.float32)
    squared_deltas = tf.square(model - y)
    loss = tf.reduce_sum(squared_deltas)
    #print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))
    #
    optimizer = tf.train.GradientDescentOptimizer(1.0)
    train = optimizer.minimize(loss)
    sess.run(init) # reset values to incorrect defaults.
    #x_train = [[[0],[0]], [[0],[1]], [[1], [0]], [[1], [1]]]
    #x_train = [[0,0], [0,1], [1, 0], [1, 1]]
    x_train = [[-1,-1], [-1,1], [1, -1], [1, 1]]
    y_train = [0, 1, 1, 0]
    for i in range(10000):
        sess.run(train, {x: x_train, y: y_train})
    #
    print(sess.run([W, b, W2, b2, loss], {x: x_train, y: y_train}))
    print(sess.run([model], {x: x_train, y: y_train}))

main()