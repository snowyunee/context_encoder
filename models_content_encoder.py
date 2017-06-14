# reference url 2개 :
# https://github.com/pathak22/context-encoder/blob/master/train.lua
# https://github.com/jazzsaxmafia/Inpainting/blob/master/src/model.py

import numpy as np
import tensorflow as tf
slim = tf.contrib.slim

def leaky_relu(x):
    # common setting negval=0.01
    return tf.maximum(0.2*x, x)

def GeneratorCNN(x, is_training, data_format, reuse):
    with tf.variable_scope("G", reuse=reuse) as vs:
        channels = [3, 64, 64, 128, 256, 512, 4000]
        with tf.variable_scope("Encoder", reuse=reuse) as vs_e:
            # encoder
            # 128 x 128 x 3 => 64 x 64 x 64 => 32 x 32 x 64 => 16 x 16 x 128
            # 8 x 8 x 256 => 4 x 4 x 512 => 1 x 1 x 4000
            for num_outputs in channels[1:]: 
                padding = "VALID" if num_outputs == 4000 else "SAME"
                x = slim.conv2d(
                        x, num_outputs=num_outputs, kernel_size=4, stride=2,
                        padding=padding, activation_fn=None,
                        data_format=data_format)
                x = slim.batch_norm(x, is_training=is_training)
                x = leaky_relu(x)
                #print("GeneratorCNN, ", num_outputs," : ",  x.get_shape().as_list())

        with tf.variable_scope("Decoder", reuse=reuse) as vs_d:
            # dencoder
            # channels : [4000, 512, 256, 128, 64, 64, 3]
            # 1 x 1 x 4000 => 4 x 4 x 512 => 8 x 8 x 256 => 
            # 128 x 128 x 3 => 64 x 64 x 64 => 32 x 32 x 64 => 16 x 16 x 128
            channels.reverse() 
            for num_outputs in channels[1:]:
                padding = "VALID" if num_outputs == 512 else "SAME"
                x = slim.conv2d_transpose(
                        x, num_outputs=num_outputs, kernel_size=4, stride=2,
                        padding=padding, activation_fn=None,
                        data_format=data_format)
                x = slim.batch_norm(x, is_training=is_training)
                if num_outputs != 3:
                    x = tf.nn.relu(x)
                #print("GeneratorCNN, ", num_outputs," : ",  x.get_shape().as_list())

    x = tf.nn.tanh(x)
    variables = tf.contrib.framework.get_variables(vs)
    return x, variables


def DiscriminatorCNN(x, is_training, data_format, reuse):
    with tf.variable_scope("D", reuse=reuse) as vs:
        # channels : [3, 64, 64, 128, 256, 512]
        # 128 x 128 x 3 => 64 x 64 x 64 => 32 x 32 x 64 => 16 x 16 x 128
        # 8 x 8 x 256 => 4 x 4 x 512 => 1
        channels = [3, 64, 64, 128, 256, 512]
        for num_outputs in channels[1:]: 
            x = slim.conv2d(
                    x, num_outputs=num_outputs, kernel_size=4, stride=2,
                    padding="SAME", activation_fn=None,
                    data_format=data_format)
            x = slim.batch_norm(x, is_training=is_training)
            x = leaky_relu(x)
            #print("DiscriminatorCNN, ", num_outputs," : ",  x.get_shape().as_list())

        dim = np.prod(x.get_shape().as_list()[1:])
        x = tf.reshape(x, [-1, dim])
        x = slim.fully_connected(x, 1, activation_fn=None)

    variables = tf.contrib.framework.get_variables(vs)
    return x, variables

