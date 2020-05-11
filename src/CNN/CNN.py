''' Implementa arquitetura da CNN3D e seu treinamento
@package CNN

@author Daniel Kim & Igor Nakamura
@date Date: 2020-04-17

@verbatim
@endverbatim 
'''

# -----------------------------
# Standard library dependencies
# -----------------------------
from typing import List, Dict

# -------------------
# Third-party imports
# -------------------
import cv2
import numpy as np
import tensorflow as tf
import tf.keras as K

def cnn3d_model(x_train):
    """ Essa função contém a arquitetura da CNN3D, baseada no paper
        3D Convolutional neural networks for human action recognition, por
        Ji et Al.
    Args:
        x_train: tensor contendo os dados de treino, no formato 
            (batch_size, dim espacial 1, dim espacial 2, dim temporal, canais)
    @return:
        dense: tensor com o output da CNN 3D
    Examples:
        >>> cnn_model(x_train)
    """
    input_shape = x_train.shape
    
    with tf.name_scope("layer_c2_s3"):
        conv1 = K.layers.conv3D(filters = 2, kernel_size = (9,7,3), 
                        input_shape = input_shape, padding = 'valid')(x_train)
        pool1 = K.layers.MaxPool3D(pool_size = (3,3,1), padding = 'valid')(conv1)
    
    with tf.name_scope("layer_c4_s5"):
        conv2 = K.layers.conv3D(filters = 3, kernel_size = (7,7,3), 
                        padding = 'valid')(pool1)
        pool2 = K.layers.MaxPool3D(pool_size = (3,3,1), padding = 'valid')(conv2)
        
    with tf.name_scope("layer_c6"):
        orig_shape = pool2.shape
        reshape_pool2 = tf.reshape(pool2, [orig_shape[0], orig_shape[1],orig_shape[2], orig_shape[3]*orig_shape[4]])
        conv3 = K.layers.conv2D(filters = 128, kernel_size = (6,4), 
                        padding = 'valid')(reshape_pool2)
        pool3 = K.layers.MaxPool3D(pool_size = (3,3,1), padding = 'valid')(conv3)
    with tf.name_scope("layer_output"):
        flat = K.layers.Flatten()(pool3)
        dense = K.layers.dense(units = 6, activation = tf.nn.softmax)(flat)
    
    return dense

def train_neural_network(x_train_data, y_train_data, x_test_data, 
                         y_test_data, learning_rate, keep_rate,
                         epochs, batch_size):


    with tf.name_scope("cross_entropy"):
        prediction = cnn3d_model(x_input)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y_input))
                              
    with tf.name_scope("training"):
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
           
    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_input, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    
    iterations = int(len(x_train_data)/batch_size) + 1
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        import datetime

        start_time = datetime.datetime.now()

        iterations = int(len(x_train_data)/batch_size) + 1
        # run epochs
        for epoch in range(epochs):
            start_time_epoch = datetime.datetime.now()
            print('Epoch', epoch, 'started', end='')
            epoch_loss = 0
            # mini batch
            for itr in range(iterations):
                mini_batch_x = x_train_data[itr*batch_size: (itr+1)*batch_size]
                mini_batch_y = y_train_data[itr*batch_size: (itr+1)*batch_size]
                _optimizer, _cost = sess.run([optimizer, cost], feed_dict={x_input: mini_batch_x, y_input: mini_batch_y})
                epoch_loss += _cost

            #  using mini batch in case not enough memory
            acc = 0
            itrs = int(len(x_test_data)/batch_size) + 1
            for itr in range(itrs):
                mini_batch_x_test = x_test_data[itr*batch_size: (itr+1)*batch_size]
                mini_batch_y_test = y_test_data[itr*batch_size: (itr+1)*batch_size]
                acc += sess.run(accuracy, feed_dict={x_input: mini_batch_x_test, y_input: mini_batch_y_test})

            end_time_epoch = datetime.datetime.now()
            print(' Testing Set Accuracy:',acc/itrs, ' Time elapse: ', str(end_time_epoch - start_time_epoch))

        end_time = datetime.datetime.now()
        print('Time elapse: ', str(end_time - start_time))
