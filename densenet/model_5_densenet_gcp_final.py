#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 12:19:11 2018

@author: vardhan
"""


#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 11:23:07 2018

@author: vardhan
"""


#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 11:45:37 2018

@author: vardhan
"""


import numpy as np
import tensorflow as tf


#defning convolution.
def conv(inputs, is_training,filters=32, kernel_size=(3, 3), strides=(1, 1),
         padding='same', activation_fn=None):
  """ Function for 2D Convolution. """
  
#  return tf.contrib.layers.convolution2d(inputs=inputs,num_outputs=filters,
#                                         kernel_size=kernel_size,stride=strides,activation_fn=activation_fn,
#                                         padding='SAME', normalizer_fn=None,
#                                         normalizer_params={'is_training': is_training},scope=scope)
#  
  return tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size,
                          strides=strides, padding=padding,
                          activation=activation_fn)

#defining transition layer
def transition_layer(inputs, trainsition_id,is_training):
  """
  Function for transition layer.
  Order: BN - ReLU - 1x1 Conv - 2x2_Avg_pool
  """
  with tf.variable_scope('transition_layer_{}'.format(trainsition_id),reuse=tf.AUTO_REUSE):
    bn = tf.layers.batch_normalization(inputs,training=is_training)
    relu = tf.nn.relu(bn)
    conv_1 = conv(relu, is_training,filters= int(relu.get_shape()[-1]), kernel_size=[1, 1])
    out = tf.layers.average_pooling2d(conv_1, [2, 2], [2, 2], padding='same')
    return out

def dense_block(inputs, num_layers, dense_block_id,is_training, growth_rate=32):
  """
  Function for Dense-Block
  For each layer in Dense-block:
    out --> BN - ReLU - 1x1_Conv (4*growth) - BN - ReLU - 3x3_Conv(growth)
    Concatenate [inputs, out]
  """
  def conv_block(inputs, conv_block_id, is_training, growth_rate=32):
    """
    Utility function for applying:
      BN - ReLU - 1x1_Conv (4*growth) - BN - ReLU - 3x3_Conv(growth)
    """
    z=1
    with tf.variable_scope('conv_block_{}_{}_{}'.format(dense_block_id, conv_block_id,z), reuse=tf.AUTO_REUSE):
      bn = tf.layers.batch_normalization(inputs,training=is_training)
      relu = tf.nn.relu(bn)
      conv_1 = conv(relu,is_training,filters= 4*growth_rate, kernel_size=[1, 1])
    z=z+1  
    with tf.variable_scope('conv_block_{}_{}_{}'.format(dense_block_id, conv_block_id,z), reuse=tf.AUTO_REUSE):
      bn_2 = tf.layers.batch_normalization(conv_1,training=is_training)
      relu_2 = tf.nn.relu(bn_2)
      out = conv(relu_2,is_training, filters=growth_rate, kernel_size=[3, 3])
      return tf.concat([inputs, out], axis=-1)

  for i in range(num_layers):
    inputs = conv_block(inputs, i,is_training, growth_rate=growth_rate)

  return inputs


def dense_net(inputs,is_training, num_blocks=4, block_layers=(6, 12, 24, 16),
              growth_rate=32, num_filters=64, num_classes=5):
  """
  Function for DenseNet-121;
  Containing 4 dense_blocks of layers - [6,12,24,16]
  Employ transition layers after each dense_block.
  """
  with tf.contrib.slim.arg_scope([tf.contrib.slim.model_variable, tf.contrib.slim.variable], device="/cpu:0"):
      with tf.variable_scope('Dense_Net', reuse=tf.AUTO_REUSE):
        conv_1 = conv(inputs,is_training,filters=num_filters,kernel_size= [7, 7], strides=[2, 2])
        bn_1 = tf.layers.batch_normalization(conv_1,training=is_training)
        relu_1 = tf.nn.relu(bn_1)
        out = tf.layers.max_pooling2d(relu_1, [3, 3], [2, 2], 'same')
    
        for i in range(num_blocks - 1):
          out = transition_layer(dense_block(out, block_layers[i], i,is_training, growth_rate), i,is_training)
    
        out = dense_block(out, block_layers[-1], num_blocks-1,is_training)
        bn = tf.layers.batch_normalization(out,training=is_training)
        relu = tf.nn.relu(bn)
        with tf.variable_scope("final_conv",reuse=tf.AUTO_REUSE):
            final_conv = conv(relu,is_training, filters=num_classes)
            bn = tf.layers.batch_normalization(final_conv,training=is_training)
            relu = tf.nn.relu(bn)
        gap = tf.layers.average_pooling2d(relu, [8, 8], [1, 1], 'valid')
        print gap.get_shape()
        out = tf.reshape(gap, [-1, num_classes])
        return out


X = tf.placeholder(tf.float32, [2, 256, 256, 1])
Y = dense_net(X,is_training=True)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  print sess.run(Y, feed_dict={X: np.random.random((2, 256, 256, 1))})
  

