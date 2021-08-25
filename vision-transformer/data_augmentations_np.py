import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.enable_eager_execution()

import numpy as np


def flip_horizontal(img):
    img = tf.image.random_flip_left_right(img)
    return img

def flip_vertical(img):
    img = tf.image.random_flip_up_down(img)
    return img

def random_colorization(img):
    img = tf.image.random_hue(img, 0.08)
    img = tf.image.random_saturation(img, 0.6, 1.6)
    img = tf.image.random_brightness(img, 0.05)
    img = tf.image.random_contrast(img, 0.7, 1.3)
    return img

def random_rotate(img):
    img = tf.keras.preprocessing.image.random_rotation(img.numpy(), 20, row_axis=0, col_axis=1, channel_axis=2)
    return tf.convert_to_tensor(img, tf.float32)

def random_shear(img):
    img = tf.keras.preprocessing.image.random_shear(img.numpy(), 25, row_axis=0, col_axis=1, channel_axis=2)
    return tf.convert_to_tensor(img, tf.float32)

def center_zoom(img):
    zoom_range_tup= (0.75,1.25)
    img= tf.keras.preprocessing.image.random_zoom(img.numpy(), zoom_range=zoom_range_tup, row_axis=1, col_axis=2, channel_axis=0, fill_mode='nearest',
    cval=0.0, interpolation_order=1
    )
    return tf.convert_to_tensor(img, tf.float32)