import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from argparse import ArgumentParser
import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import os
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from model_tf1_jaidev import VisionTransformer
import numpy as np

from new_data_augmentations import random_hue_saturation, random_brightness_contrast, flip_horizontal, rotate_20, rotate_30, flip_vertical, random_zoom_crop, grayscale, random_jpeg_noise

AUTOTUNE = tf.data.experimental.AUTOTUNE

#Run on GPU0
# os.environ["DML_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == "__main__":
    
    #tf.debugging.set_log_device_placement(True) 
    tf.enable_eager_execution() 

    IMAGE_SIZE= 32
    NUMBER_OF_CLASSES= 10
    PATCH_SIZE= 4
    PATCH_STRIDE=4
    NUMBER_OF_LAYERS=8
    EMBEDDING_DIM=64
    NUM_HEADS= 8
    MLP_HIDDEN_DIM= 256
    LEARNING_RATE= 0.001 #3e-4
    BATCH_SIZE= 512
    EPOCHS= 100
    PATIENCE= 10

    (X_train, y_train) , (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.15, shuffle=True)
    X_train_aug=list()

    X_train_aug=list()
    for img in X_train:
        img_aug = tf.keras.preprocessing.image.random_rotation(img, 20, row_axis=0, col_axis=1, channel_axis=2)
        X_train_aug.append(img_aug)
        img_aug = tf.keras.preprocessing.image.random_shear(img, 25, row_axis=0, col_axis=1, channel_axis=2)
        X_train_aug.append(img_aug)

    X_train_aug= np.asarray(X_train_aug)
    #X_train= np.concatenate((X_train, X_train_aug), axis=0)
    #y_train= np.concatenate((y_train, y_train, y_train), axis=0)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    validation_dataset = tf.data.Dataset.from_tensor_slices((X_validate, y_validate))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    def cast_to_float(x,y):
        return tf.cast(x, tf.float32), tf.cast(y, tf.float32)

    train_dataset = train_dataset.map(cast_to_float)
    validation_dataset = validation_dataset.map(cast_to_float)

    def get_train_dataset(train_dataset):
        augmentations = [random_hue_saturation, random_brightness_contrast, flip_horizontal, flip_vertical]
        for aug in augmentations:
            train_dataset_aug = train_dataset.map(lambda x, y: (tf.cond(tf.random_uniform([], 0, 1) > 0.5, lambda: aug(x), lambda: x), y), num_parallel_calls=AUTOTUNE)
        
        train_dataset= train_dataset.concatenate(train_dataset_aug)

        train_dataset= train_dataset.shuffle(10000, reshuffle_each_iteration= True)
        train_dataset= train_dataset.cache()
        train_dataset=train_dataset.batch(BATCH_SIZE)
        train_dataset=train_dataset.prefetch(AUTOTUNE)
        
        return train_dataset

    model = VisionTransformer(
        image_size= IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        patch_stride=PATCH_STRIDE,
        num_layers=NUMBER_OF_LAYERS,
        num_classes=NUMBER_OF_CLASSES,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        mlp_hidden_dim=MLP_HIDDEN_DIM,
    )
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-07),
        metrics=[
        tf.keras.metrics.SparseCategoricalAccuracy(name="Top-1-accuracy"),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(3, name="Top-3-accuracy"),
        ],
    )
    
file_path= './saved_models/Model_tf1_cifar10_with_data_augmentation'
checkpoint = ModelCheckpoint(file_path, monitor='val_Top-1-accuracy', verbose=1, save_best_only=True, mode='max')
reduce_on_plateau = ReduceLROnPlateau(monitor="val_Top-1-accuracy", mode="max", factor=0.5, patience=PATIENCE, verbose=1,min_lr=0.00002)
callbacks_list = [checkpoint, reduce_on_plateau]


model.fit(
    get_train_dataset(train_dataset),
    validation_data=(X_validate,y_validate),
    epochs=EPOCHS,
    callbacks=callbacks_list,
)

#Compute Metrics on Test Set
test_metrics= model.evaluate(X_test, y_test, 128)
print(test_metrics)
