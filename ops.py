import tensorflow as tf

def conv_layer(name, X, in_channels, out_filters, ksize, stride, padding="SAME", trainable=True,
               activator=tf.nn.elu,
               weight_initializer=tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform"),
               bias_initializer=tf.constant_initializer(0.0),
               groups=1, spectral_norm=0,
               with_bn=False, momentum=0.99, epsilon=1e-5):
    if not hasattr(ksize, "__len__"):
        ksize = [ksize, ksize]
    if not hasattr(stride, "__len__"):
        stride = [stride, stride]
    if len(X.shape) == 2:
        X = tf.reshape(X, [-1, 1, X.shape[1], 1])
        in_channels = 1
    else:
        assert(len(X.shape) == 4)
    with tf.variable_scope(name):
        w_shape = [ksize[0], ksize[1], in_channels//groups, out_filters]
        w = tf.get_variable("weight", w_shape, tf.float32,
                            weight_initializer, trainable=trainable)
        if spectral_norm > 0:
            # Spectral Normalization for Generative Adversarial Networks (ICLR 2018)
            l2_norm = lambda _: tf.sqrt(tf.reduce_sum(tf.square(_))) 
            w_ = tf.reshape(w, [-1, w.shape[-1]])
            w_T = tf.transpose(w_)
            u = tf.get_variable("u", [1, w.shape[-1]], tf.float32,
                        tf.truncated_normal_initializer(), trainable=False)
            u_ = u
            for _ in range(spectral_norm):
                v = tf.matmul(u_, w_T)
                v_ = v / (l2_norm(v) + 1e-9)
                u_ = tf.matmul(v_, w_)
                u_ /= l2_norm(u_) + 1e-9
            sigma = tf.matmul(tf.matmul(v_, w_), tf.transpose(u_))
            w_ /= sigma
            with tf.control_dependencies([u.assign(u_)]):
                w = tf.reshape(w_, w_shape)
        if groups == 1:
            Y = tf.nn.conv2d(X, w, [1, stride[0], stride[1], 1], padding)
        else:
            xs = tf.split(X, groups, axis=-1)
            ws = tf.split(w, groups, axis=-1)
            Y = tf.concat([
                tf.nn.conv2d(_, __, [1, stride[0], stride[1], 1], padding, name="conv_group")
                for _, __ in zip(xs, ws)
            ], axis=-1)
        if with_bn:
            Y = bn_layer('batch_norm', Y, momentum=momentum, epsilon=epsilon,
                    activator=activator, trainable=trainable)
        else:
            b = tf.get_variable("bias", [out_filters], tf.float32,
                                bias_initializer,
                                trainable=trainable)
            Y = tf.add(Y, b)
            if activator is not None:
                Y = activator(Y)
    return Y


def max_pool(name, X, ksize, stride, padding="SAME"):
    with tf.variable_scope(name):
        ksize = [1, ksize[0], ksize[1], 1] if hasattr(ksize, "__len__") else [1, ksize, ksize, 1]
        stride = [1, stride[0], stride[1], 1] if hasattr(stride, "__len__") else [1, stride, stride, 1]
        Y = tf.nn.max_pool(X, ksize, stride, padding)
    return Y

def bn_layer(name, X, momentum=0.99, epsilon=1e-5, trainable=True,
            activator=tf.nn.elu,):
    Y = tf.layers.batch_normalization(X, momentum=momentum, epsilon=epsilon,
                                    center=True, scale=True, renorm=True,
                                    trainable=True,
                                    training=trainable,
                                    name=name)
    if activator is not None:
        Y = activator(Y)
    return Y