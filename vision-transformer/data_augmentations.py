import tensorflow.compat.v1 as tf
import math
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

"""
def random_rotate(img):
    img= tf.contrib.image.rotate(img, 30 * math.pi / 180, interpolation='BILINEAR')
    return tf.convert_to_tensor(img, tf.float32)
"""

def grayscale(img):
    img = tf.image.rgb_to_grayscale(img)
    return tf.convert_to_tensor(img, tf.float32)

def saturate(img):
    img = tf.image.adjust_saturation(img, 3)
    return tf.convert_to_tensor(img, tf.float32) 

"""
def center_zoom(img):
    img= tf.image.central_crop(img, 0.6)
    img= tf.expand_dims(img, axis=0)
    img= tf.image.resize_bilinear(img, (32,32))
    img=tf.squeeze(img)
    return tf.convert_to_tensor(img, tf.float32)
"""