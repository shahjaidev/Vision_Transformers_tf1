import os
from argparse import ArgumentParser
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
from argparse import ArgumentParser
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from model_tf1 import VisionTransformer

tf.enable_eager_execution()
AUTOTUNE = tf.data.experimental.AUTOTUNE

#Run on GPU1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == "__main__":
 
    parser = ArgumentParser()
    parser.add_argument("--logdir", default="logs")
    """
    parser.add_argument("--image-size", default=32, type=int)
    parser.add_argument("--patch-size", default=4, type=int)
    parser.add_argument("--num-layers", default=4, type=int)
    parser.add_argument("--d-model", default=64, type=int)
    parser.add_argument("--num-heads", default=4, type=int)
    parser.add_argument("--mlp-dim", default=128, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--batch-size", default=1024, type=int)
    parser.add_argument("--epochs", default=500, type=int)
    args = parser.parse_args()
    """
    IMAGE_SIZE= 32
    NUMBER_OF_CLASSES= 100

    PATCH_SIZE= 4
    NUMBER_OF_LAYERS=4
    D_MODEL=64
    NUM_HEADS= 8
    MLP_DIM= 128
    LEARNING_RATE= 0.001 #3e-4
    WEIGHT_DECAY= 1e-4
    BATCH_SIZE= 1024
    EPOCHS= 500
    PATIENCE= 3

    (X_train, y_train) , (X_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode="fine")
    X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.2, random_state=15)


    datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zca_epsilon=1e-06,
    zca_whitening=True,)

    datagen.fit(X_train)

    
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    validation_dataset = tf.data.Dataset.from_tensor_slices((X_validate, y_validate))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    def cast_to_float(x,y):
        return tf.cast(x, tf.float32), tf.cast(y, tf.float32)
    train_dataset = train_dataset.map(cast_to_float)
    validation_dataset = validation_dataset.map(cast_to_float)
    

    train_dataset = (
        train_dataset
        .cache()
        .shuffle(10000, reshuffle_each_iteration= True)
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )
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
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        mlp_dim=MLP_DIM,
        channels=3,
        dropout=0.2,
    )
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tfa.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY),
        metrics=["accuracy"],
    )
    
file_path= './saved_models/Model_1'
checkpoint = ModelCheckpoint(file_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
reduce_on_plateau = ReduceLROnPlateau(monitor="val_accuracy", mode="max", factor=0.5, patience=PATIENCE, verbose=1,min_lr=0.00002)
callbacks_list = [checkpoint, reduce_on_plateau]

"""
model.fit_generator(datagen.flow(X_train, y_train,batch_size=BATCH_SIZE), validation_data= validation_dataset,
                    steps_per_epoch=len(X_train) / BATCH_SIZE, epochs=EPOCHS,
                    callbacks=callbacks_list)
"""

model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=callbacks_list,
)
model.save_weights(os.path.join(args.logdir, "vit"))
