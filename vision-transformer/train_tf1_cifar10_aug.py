import os
from argparse import ArgumentParser
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from model_tf1 import VisionTransformer
import numpy as np

from old_data_augmentations import random_colorization,flip_horizontal, flip_vertical, random_zoom_crop, grayscale, random_jpeg_noise

#from tensorflow.keras.preprocessing.image import random_brightness, random_shift, random_zoom, random_shear, random_rotation

AUTOTUNE = tf.data.experimental.AUTOTUNE

#Run on GPU1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == "__main__":
    
    tf.enable_eager_execution()

    IMAGE_SIZE= 32
    NUMBER_OF_CLASSES= 10
    PATCH_SIZE= 4
    NUMBER_OF_LAYERS=8
    EMBEDDING_DIM=64
    NUM_HEADS= 8
    MLP_DIM_L1= 1024
    MLP_DIM_L2= 512
    LEARNING_RATE= 0.001 #3e-4
    BATCH_SIZE= 128
    EPOCHS= 100#300
    PATIENCE= 10

    (X_train, y_train) , (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.15, shuffle=True)

    X_train= tf.image.resize(X_train, [72,72], method=tf.image.ResizeMethod.BILINEAR)
    X_train_aug=list()

    X_train_aug=list()
    for img in X_train:
        img_aug = tf.keras.preprocessing.image.random_rotation(img.numpy(), 20, row_axis=0, col_axis=1, channel_axis=2)
        img_aug = tf.keras.preprocessing.image.random_shear(img.numpy(), 25, row_axis=0, col_axis=1, channel_axis=2)
        X_train_aug.append(img_aug)


    print(X_train_aug)

    datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range= [0.75,1.25],
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zca_epsilon=1e-06,
    zca_whitening=True)


    def data_augmentation(img):
        print("img")
        print(img)
        #img = tf.keras.preprocessing.image.random_rotation(img, 20, row_axis=0, col_axis=1, channel_axis=2)
        img = tf.keras.preprocessing.image.random_shear(img, 30, row_axis=0, col_axis=1, channel_axis=2)
        img = tf.keras.preprocessing.image.random_brightness(img, (0.2, 1.6))
        img = tf.keras.preprocessing.image.random_zoom(img, (0.75,1.25))
        img= tf.keras.preprocessing.image.random_shift(img, 0.2, 0.2, row_axis=0, col_axis=1, channel_axis=2,fill_mode='nearest', cval=0.0, interpolation_order=1)

        return img

    #X_train= X_train.astype(np.float32)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    #train_dataset = train_dataset.batch(16).map(lambda x, y: (data_augmentation(x), y))
    #train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x), y))

    validation_dataset = tf.data.Dataset.from_tensor_slices((X_validate, y_validate))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    def cast_to_float(x,y):
        return tf.cast(x, tf.float32), tf.cast(y, tf.float32)
    train_dataset = train_dataset.map(cast_to_float)
    validation_dataset = validation_dataset.map(cast_to_float)

    def get_train_dataset(train_dataset):
        #train_dataset = train_dataset.map(lambda x, y: (tf.image.per_image_standardization(x), y), num_parallel_calls=AUTOTUNE)
        augmentations = [random_colorization,flip_horizontal, flip_vertical]
        for aug in augmentations:
            #train_dataset = train_dataset.map(lambda x, y: (tf.image.resize(x, [72,72], method=tf.image.ResizeMethod.BILINEAR), y), num_parallel_calls=AUTOTUNE)
            train_dataset = train_dataset.map(lambda x, y: (tf.cond(tf.random_uniform([], 0, 1) > 0.9, lambda: aug(x), lambda: x), y), num_parallel_calls=AUTOTUNE)
           
        train_dataset = (
            train_dataset
            .cache()
            .shuffle(10000, reshuffle_each_iteration= True)
            .batch(BATCH_SIZE)
            .prefetch(AUTOTUNE)
        )
        
        return train_dataset

    validation_dataset = (
        validation_dataset
        .cache()
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )

    model = VisionTransformer(
        image_size= IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        num_layers=NUMBER_OF_LAYERS,
        num_classes=NUMBER_OF_CLASSES,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        mlp_dim_l1=MLP_DIM_L1,
        mlp_dim_l2=MLP_DIM_L2,
        channels=3,
        dropout=0.25,
    )
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-07),
        metrics=[
        tf.keras.metrics.SparseCategoricalAccuracy(name="Top-1-accuracy"),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(3, name="Top-3-accuracy"),
        ],
        run_eagerly=True,
    )
    
file_path= './saved_models/Model_tf1_cifar10_aug'
checkpoint = ModelCheckpoint(file_path, monitor='val_Top-1-accuracy', verbose=1, save_best_only=True, mode='max')
reduce_on_plateau = ReduceLROnPlateau(monitor="val_Top-1-accuracy", mode="max", factor=0.5, patience=PATIENCE, verbose=1,min_lr=0.00002)
callbacks_list = [checkpoint, reduce_on_plateau]
                 
model.fit(
    get_train_dataset(train_dataset),
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=callbacks_list,
)

#Compute Metrics on Test Set

test_metrics= model.evaluate(X_test, y_test, 32)
print(test_metrics)
