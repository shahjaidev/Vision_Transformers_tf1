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

def random_hue_saturation(img):
    img = tf.image.random_hue(img, 0.08)
    img = tf.image.random_saturation(img, 2, 5)

    return img

def random_brightness_contrast(img):
    
    #Adjusts contrast using a contrast_factor chosen randomly from the range [-0.25,8]
    img = tf.image.random_contrast(img, 0.25, 0.8)

    # Adjusts brightness using a delta chosen randomly from the range [-0.4,0.4]
    img = tf.image.random_brightness(img, 0.4)
    return img


def grayscale(img):
    #Converts RGB images to grayscale but changes channel dimensions to 1. 
    img = tf.image.rgb_to_grayscale(img)
    return img

def rotate_20(img):
    #Rotates image by pi/9 radians , that is, 20 degrees
    img= tf.contrib.image.rotate(img, 20 * np.pi / 180, interpolation='BILINEAR')
    return img

def rotate_30(img):
    #Rotates image by pi/6 radians , that is, 30 degrees
    img= tf.contrib.image.rotate(img, 30 * np.pi / 180, interpolation='BILINEAR')
    return img

def random_jpeg_noise(img):
    #min_jpeg_quality: 60, max_jpeg_quality: 100
    img= tf.image.random_jpeg_quality(img, 60, 100)
    return img

def random_zoom_crop(img: tf.Tensor) -> tf.Tensor:
    # Generate 20 crop settings, ranging from a 1% to 20% crop.
    scales = list(np.arange(0.8, 1.0, 0.01))
    boxes = np.zeros((len(scales), 4))

    for i, scale in enumerate(scales):
        x1 = y1 = 0.5 - (0.5 * scale)
        x2 = y2 = 0.5 + (0.5 * scale)
        boxes[i] = [x1, y1, x2, y2]

    def random_crop(img):
        # Create different crops for an image
        crops = tf.image.crop_and_resize([img], boxes=boxes, box_ind=np.zeros(len(scales)), crop_size=(32, 32))
        # Return a random crop
        return crops[tf.random_uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32)]


    choice = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)

    return random_crop(img)