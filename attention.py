import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
def conv2d(input_layer, input_channels, output_channels, scope, kernel_size=3, stride=1, padding='SAME', bias=False):
    strides = [1, stride, stride, 1]
    with tf.variable_scope(scope):
        conv_filter = tf.get_variable(
            'weight',
            shape = [kernel_size, kernel_size, input_channels, output_channels],
            dtype = tf.float32,
            initializer = tf.contrib.layers.variance_scaling_initializer(),
            regularizer = tf.contrib.layers.l2_regularizer(scale=0.0002)
        )
        conv = tf.nn.conv2d(input_layer, conv_filter, strides, padding)

        if bias:
            bias = tf.get_variable(
                'bias',
                shape = [output_channels],
                dtype = tf.float32,
                initializer = tf.constant_initializer(0.0)
            )
            output_layer = tf.nn.bias_add(conv, bias)
            output_layer = tf.reshape(output_layer, conv.get_shape())
        else:
            output_layer = conv

        return output_layer

def NonLocalBlock(input_x, output_channels, sub_sample=False, is_bn=False, is_training=True, scope="NonLocalBlock"):
    batch_size, height, width, in_channels = input_x.get_shape().as_list()
    with tf.variable_scope(scope):
        with tf.variable_scope("g"):
            g = tf.layers.conv2d(inputs=input_x, filters=output_channels, kernel_size=1, strides=1, padding="same", name="g_conv")
            if sub_sample:
                g = tf.layers.max_pooling2d(inputs=g, pool_size=2, strides=2, padding="valid", name="g_max_pool")
                #print(g.shape)

        with tf.variable_scope("phi"):
            phi = tf.layers.conv2d(inputs=input_x, filters=output_channels, kernel_size=1, strides=1, padding="same", name="phi_conv")
            if sub_sample:
                phi = tf.layers.max_pooling2d(inputs=phi, pool_size=2, strides=2, padding="valid", name="phi_max_pool")
                #print(phi.shape)

        with tf.variable_scope("theta"):
            theta = tf.layers.conv2d(inputs=input_x, filters=output_channels, kernel_size=1, strides=1, padding="same", name="theta_conv")
            #print(theta.shape)

        #g_x = tf.reshape(g, [-1, output_channels, height * width])
        g_x = tf.reshape(g, [-1, height * width, output_channels])
        #g_x = tf.transpose(g_x, [0, 2, 1])
        #print(g_x.shape)

        phi_x = tf.reshape(phi, [-1, output_channels, height * width])
        #print(phi_x.shape)

        #theta_x = tf.reshape(theta, [-1, output_channels, height * width])
        #theta_x = tf.transpose(theta_x, [0, 2, 1])
        theta_x = tf.reshape(theta, [-1, height * width, output_channels])
        #print(theta_x.shape)

        f = tf.matmul(theta_x, phi_x)
        f_softmax = tf.nn.softmax(f, -1)
        y = tf.matmul(f_softmax, g_x)

        y = tf.reshape(y, [-1, height, width, output_channels])

        with tf.variable_scope("w"):
            w_y = tf.layers.conv2d(inputs=y, filters=in_channels, kernel_size=1, strides=1, padding="same", name="w_conv")
            if is_bn:
                w_y= tf.layers.batch_normalization(w_y, axis=3, training=is_training)    ### batch_normalization
        z = input_x + w_y

        return z

# CBAM
def channel_attention(input_feature, name, ratio):
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
    bias_initializer = tf.constant_initializer(value=0.0)

    with tf.variable_scope(name):
        channel = input_feature.get_shape()[-1]
        avg_pool = tf.reduce_mean(input_feature, axis=[1, 2], keep_dims=True)

        assert avg_pool.get_shape()[1:] == (1, 1, channel)
        avg_pool = tf.layers.dense(inputs=avg_pool,
                                   units=channel // ratio,
                                   activation=tf.nn.relu,
                                   kernel_initializer=kernel_initializer,
                                   bias_initializer=bias_initializer,
                                   name='mlp_0',
                                   reuse=None)    # None
        assert avg_pool.get_shape()[1:] == (1, 1, channel // ratio)
        avg_pool = tf.layers.dense(inputs=avg_pool,
                                   units=channel,
                                   kernel_initializer=kernel_initializer,
                                   bias_initializer=bias_initializer,
                                   name='mlp_1',
                                   reuse=None)     # None
        assert avg_pool.get_shape()[1:] == (1, 1, channel)

        max_pool = tf.reduce_max(input_feature, axis=[1, 2], keep_dims=True)
        assert max_pool.get_shape()[1:] == (1, 1, channel)
        max_pool = tf.layers.dense(inputs=max_pool,
                                   units=channel // ratio,
                                   activation=tf.nn.relu,
                                   name='mlp_0',
                                   reuse=True)
        assert max_pool.get_shape()[1:] == (1, 1, channel // ratio)
        max_pool = tf.layers.dense(inputs=max_pool,
                                   units=channel,
                                   name='mlp_1',
                                   reuse=True)
        assert max_pool.get_shape()[1:] == (1, 1, channel)

        scale = tf.sigmoid(avg_pool + max_pool, 'sigmoid')

    return input_feature * scale

def spatial_attention(input_feature, name):
    kernel_size = 7      # default
    kernel_size = 1
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
    with tf.variable_scope(name):
        avg_pool = tf.reduce_mean(input_feature, axis=[3], keep_dims=True)
        assert avg_pool.get_shape()[-1] == 1
        max_pool = tf.reduce_max(input_feature, axis=[3], keep_dims=True)
        assert max_pool.get_shape()[-1] == 1
        concat = tf.concat([avg_pool, max_pool], 3)
        assert concat.get_shape()[-1] == 2

        concat = tf.layers.conv2d(concat,
                                  filters=1,
                                  kernel_size=[kernel_size, kernel_size],
                                  strides=[1, 1],
                                  padding="same",
                                  activation=None,
                                  kernel_initializer=kernel_initializer,
                                  use_bias=False,
                                  name='conv')
        assert concat.get_shape()[-1] == 1
        concat = tf.sigmoid(concat, 'sigmoid')

    return input_feature * concat

