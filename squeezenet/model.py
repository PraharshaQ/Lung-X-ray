
import tensorflow as tf
num_classes = 15

def max_pool(x, ksize=(2, 2), stride=(2, 2)):
  return tf.nn.max_pool(x, ksize=[1, ksize[0], ksize[1], 1],
                        strides=[1, stride[0], stride[1], 1], padding='VALID')


def avg_pool(x, ksize=(8, 8), stride=(1, 1)):                                 # Understand why kernel_size = (16,16)
  return tf.nn.avg_pool(x, ksize=[1, ksize[0], ksize[1], 1],
                        strides=[1, stride[0], stride[1], 1], padding='VALID')


def conv(inputs, num_ouputs, kernel_size, is_training, scope, stride=[1,1], activation_fn=tf.nn.relu):
  return tf.contrib.layers.convolution2d(inputs=inputs,num_outputs=num_ouputs,
                                         kernel_size=kernel_size,stride=stride,activation_fn=activation_fn,
                                         padding='SAME', normalizer_fn=tf.contrib.layers.batch_norm,
                                         normalizer_params={'is_training': is_training},scope=scope)


def fire(inputs, n_hidden, f_ind, is_training,reuse):
  with tf.variable_scope('f{}-sq-1'.format(f_ind)) as scope:
    f1_sq_1 = conv(inputs, n_hidden/4, [1,1], is_training, scope)
  with tf.variable_scope('f{}-e-1'.format(f_ind)) as scope:
    f1_e_1 = conv(f1_sq_1, n_hidden, [1,1], is_training, scope, activation_fn=None)
  with tf.variable_scope('f{}-e-3'.format(f_ind)) as scope:
    f1_e_3 = conv(f1_sq_1, n_hidden, [3,3], is_training, scope, activation_fn=None)
  if f_ind % 2 != 0:
    return tf.nn.relu(tf.concat([f1_e_1, f1_e_3],axis=3))
  else:
    return max_pool(tf.nn.relu(tf.add(tf.concat([f1_e_1,f1_e_3], axis=3), inputs)))


n_hidden_1 = 32
n_hidden_2 = 32
n_hidden_3 = 64
n_hidden_4 = 64
n_hidden_5 = 128
n_hidden_6 = 128
n_hidden_7 = 256
n_hidden_8 = 256
n_hidden_9 = 512
n_hidden_10 = 512


def model(x,is_training,reuse):
    x=tf.reshape(x,[-1,256,256,1])
    f_ind = 1
    with tf.contrib.slim.arg_scope([tf.contrib.slim.model_variable, tf.contrib.slim.variable], device="/cpu:0"):
        with tf.variable_scope('2D-Conv-1',reuse) as scope:
            layer_1 = conv(x, n_hidden_1, [3,3], is_training, scope)
        f_1 = fire(layer_1, n_hidden_1, f_ind, is_training, reuse); f_ind += 1;
        f_2 = fire(f_1, n_hidden_2, f_ind, is_training, reuse); f_ind += 1;
        f_3 = fire(f_2, n_hidden_3, f_ind, is_training, reuse); f_ind += 1;
        f_4 = fire(f_3, n_hidden_4, f_ind, is_training, reuse); f_ind += 1;
        f_5 = fire(f_4, n_hidden_5, f_ind, is_training, reuse); f_ind += 1;
        f_6 = fire(f_5, n_hidden_6, f_ind, is_training, reuse); f_ind += 1;
        f_7 = fire(f_6, n_hidden_7, f_ind, is_training, reuse); f_ind += 1;
        f_8 = fire(f_7, n_hidden_8, f_ind, is_training, reuse); f_ind += 1;
        f_9 = fire(f_8, n_hidden_9, f_ind, is_training, reuse); f_ind += 1;
        f_10 = fire(f_9, n_hidden_10, f_ind, is_training, reuse); f_ind += 1;
        with tf.variable_scope('Final_Conv') as scope:
            final_conv = conv(f_10, num_classes, [1,1], is_training, scope)
            final_pool = avg_pool(final_conv)
        logits = tf.reshape(final_pool,[-1, num_classes], name='logits_node') # Note: name 'logits_node'
        logits=tf.cast(logits,tf.float32)
        return logits

#pp = model(tf.random_normal((32,256,256,1)),is_training=True,reuse=tf.AUTO_REUSE)

