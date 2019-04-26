from ops import *

__all__ = [
    "classification_loss",
    "vgg16", "vgg19"
]

def classification_loss(X, logits, Y=None):
    if Y is None:
        Y = tf.placeholder(tf.int32, [None], name="label")
    with tf.name_scope("loss"):
        xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=Y)
        loss = tf.reduce_mean(xent)
    with tf.name_scope("acc"):
        acc = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(logits, axis=1, output_type=tf.int32), Y), tf.float32))/tf.cast(tf.shape(X)[0], tf.float32)
    return Y, acc, loss

def vgg(trainable, n_classes, layers, x, drop_rate=None):
    for i, n_layer in enumerate(layers):
        for j in range(n_layer):
            x = conv_layer("conv{}_{}".format(i+1, j+1), x, x.shape[-1], min(512, 64*(2**i)), 3, 1,
                    with_bn=True, trainable=trainable)
        x = max_pool("pool{}".format(i+1), x, 2, 2)

    x = conv_layer("fc1", x, x.shape[-1], 4096, x.shape[1:3], 1, padding="VALID",
               with_bn=True, trainable=trainable)
    if drop_rate is not None:
        x = tf.nn.dropout(x, rate=drop_rate)
    x = conv_layer("fc2", x, 4096, 4096, 1, 1,
               with_bn=True, trainable=trainable)
    if drop_rate is not None:
        x = tf.nn.dropout(x, rate=drop_rate)
    x = conv_layer("fc3", x, 4096, n_classes, 1, 1,
               with_bn=False, activator=None, trainable=trainable)

    logits = tf.squeeze(x, [1,2])
    return logits

def vgg16(trainable, x, n_classes, drop_rate=None):
    return vgg(trainable, n_classes, [2, 2, 3, 3, 3], x, drop_rate)

def vgg19(trainable, x, n_classes, drop_rate=None):
    return vgg(trainable, n_classes, [2, 2, 4, 4, 4], x, drop_rate)
