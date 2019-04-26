import os, random, functools
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
import numpy as np

def load_data(folder, exclude=None, include=None, include_ext=None, exclude_ext=None, sort_fn=sorted):
    data = {}
    for label in sort_fn(os.listdir(folder)):
        if include is not None and label not in include:
            continue
        if exclude is not None and label in exclude:
            continue
        path = os.path.join(folder, label)
        if os.path.isdir(path):
            images = []
            for f in os.listdir(path):
                ext = f.split(".")[-1].lower()
                if include_ext is not None and ext not in include_ext:
                    continue
                if exclude_ext is not None and ext in exclude_ext:
                    continue
                images.append(os.path.join(path, f))
            data[label] = images
            if len(images) == 0:
                print("[WARN] Detected empty folder `{}`".format(label))
    return data

def load_im(image_file):
    im = tf.read_file(image_file)
    im = tf.io.decode_image(im, channels=3)
    im.set_shape([None, None, 3])
    return im

def grayscale(im):
    return tf.image.rgb_to_grayscale(im)

def random_grayscale(im):
    def convert(im, flag):
        if flag == 0:
            im = tf.image.rgb_to_grayscale(im)
            im = tf.stack([im, im, im], axis=-1)
        return im
    choice = tf.random_uniform([], maxval=4, dtype=tf.int32)
    im = control_flow_ops.merge([
        convert(control_flow_ops.switch(im, tf.equal(choice, flag))[1], flag)
        for flag in range(2)
    ])[0]
    return im

def random_resize(im, size):
    choice = tf.random_uniform([], maxval=4, dtype=tf.int32)
    im = control_flow_ops.merge([
        tf.image.resize_images(control_flow_ops.switch(im, tf.equal(choice, method))[1], size, method)
        for method in range(4)
    ])[0]
    return im

def to_float(im):
    return tf.cast(im, tf.float32) / 255.0

def normalize(im, mean, std):
    return (im - mean) / std

def per_image_standardization(im):
    return tf.image.per_image_standardization(im)

def random_horizontal_flip(im):
    return tf.image.random_flip_left_right(im)

def random_vertical_flip(im):
    return tf.image.random_flip_up_down(im)

def random_flip(im):
    return random_vertical_flip(random_horizontal_flip(im))

def random_hue(im, delta=0.2):
    return tf.image.random_hue(im, delta)

def random_contrast(im, lower=0.8, upper=1.25):
    return tf.image.random_contrast(im, lower, upper)

def random_brightness(im, max_delta=0.2):
    return tf.image.random_brightness(im, max_delta)

def random_saturation(im, lower=0.8, upper=1.25):
    return tf.image.random_saturation(im, lower, upper)

def random_distort_color(im):
    def distort(im, order):
        if order == 0:
            im = random_contrast(random_hue(random_saturation(random_brightness(im))))
        elif order == 1:
            im = random_hue(random_contrast(random_brightness(random_saturation(im))))
        elif order == 2:
            im = random_saturation(random_brightness(random_hue(random_contrast(im))))
        else:
            im = random_brightness(random_contrast(random_saturation(random_hue(im))))
        return im
    choice = tf.random_uniform([], maxval=4, dtype=tf.int32)
    im = control_flow_ops.merge([
        distort(control_flow_ops.switch(im, tf.equal(choice, order))[1], order)
        for order in range(4)
    ])[0]
    return im

def random_crop(im, bbox=None, area_range=[0.25, 1.0]):
    channel = im.shape[-1]
    if bbox is None:
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
            tf.shape(im), bbox, area_range=area_range, use_image_if_no_bounding_boxes=True)
    im = tf.slice(im, bbox_begin, bbox_size)
    im.set_shape([None, None, channel])
    return im

def resize(im, size):
    return tf.image.resize_images(im, size)

def random_op(x, ops):
    def choose(x, case):
        return ops[case](x)
    choice = tf.random_uniform([], maxval=len(ops), dtype=tf.int32)
    im = control_flow_ops.merge([
        choose(control_flow_ops.switch(x, tf.equal(choice, case))[1], case)
        for case in range(len(ops))
    ])[0]
    return im

def gauss_noise(im, mean=0.0, std=0.05):
    noise = tf.random_normal(shape=tf.shape(im), mean=mean, stddev=std, dtype=tf.float32)
    return tf.add(im, noise)

def sharpen(im):
    with tf.device("/cpu:0"):
        kernel = tf.constant([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], tf.float32)
        kernel = tf.stack([kernel]*im.shape[-1], axis=-1)
        kernel = tf.expand_dims(kernel, axis=-1)
        expand = len(im.shape) != 4
        if expand: im = tf.expand_dims(im, axis=0)
        im = tf.nn.depthwise_conv2d(im, kernel, [1,1,1,1], "SAME")
        return tf.squeeze(im, axis=0) if expand else im

def unsharpen(im):
    with tf.device("/cpu:0"):
        kernel = np.array([
            [1, 4, 6, 4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, -476, 24, 6],
            [4, 16, 24, 16, 4],
            [1, 4, 6, 4, 1]
        ], np.float32) / -256.0
        kernel = tf.stack([kernel]*im.shape[-1], axis=-1)
        kernel = tf.expand_dims(kernel, axis=-1)
        expand = len(im.shape) != 4
        if expand: im = tf.expand_dims(im, axis=0)
        im = tf.nn.depthwise_conv2d(im, kernel, [1,1,1,1], "SAME")
        return tf.squeeze(im, axis=0) if expand else im

def avg_blur(im, size=(3, 7)):
    with tf.device("/cpu:0"):
        if hasattr(size, "__len__"):
            size = tf.random_uniform([], minval=min(size), maxval=max(size), dtype=tf.int32)
        kernel = tf.ones([size, size], dtype=tf.float32)
        kernel /= tf.reduce_sum(kernel)
        kernel = tf.stack([kernel]*im.shape[-1], axis=-1)
        kernel = tf.expand_dims(kernel, axis=-1)
        expand = len(im.shape) != 4
        if expand: im = tf.expand_dims(im, axis=0)
        im = tf.nn.depthwise_conv2d(im, kernel, [1,1,1,1], "SAME")
        return tf.squeeze(im, axis=0) if expand else im

def gauss_blur(im, size=(3, 7), std=(1.5, 2.0)):
    with tf.device("/cpu:0"):
        if hasattr(size, "__len__"):
            size = tf.random_uniform([], minval=min(size), maxval=max(size), dtype=tf.int32)
        if hasattr(std, "__len__"):
            var = tf.random_uniform([], minval=min(std)**2, maxval=max(std)**2, dtype=tf.float32)
        else:
            var = tf.square(std)
        bound = tf.range(-size//2+1, size//2+1)
        x, y = tf.meshgrid(bound, bound)
        xx = tf.square(tf.cast(x, tf.float32))
        yy = tf.square(tf.cast(y, tf.float32))
        kernel = tf.exp(-0.5*(xx+yy)/var)
        kernel /= tf.reduce_sum(kernel)
        kernel = tf.stack([kernel]*im.shape[-1], axis=-1)
        kernel = tf.expand_dims(kernel, axis=-1)
        expand = len(im.shape) != 4
        if expand: im = tf.expand_dims(im, axis=0)
        im = tf.nn.depthwise_conv2d(im, kernel, [1,1,1,1], "SAME")
        return tf.squeeze(im, axis=0) if expand else im

 
