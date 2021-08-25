import os
from argparse import ArgumentParser
import tensorflow as tf

import tensorflow_addons as tfa

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from model import VisionTransformer

AUTOTUNE = tf.data.experimental.AUTOTUNE

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
    IMAGE_SIZE= 72
    NUMBER_OF_CLASSES= 10

    PATCH_SIZE= 6
    NUMBER_OF_LAYERS=4
    D_MODEL=64
    NUM_HEADS= 8
    MLP_DIM= 128
    LEARNING_RATE= 0.001 #3e-4
    WEIGHT_DECAY= 1e-4
    BATCH_SIZE= 4096
    EPOCHS= 500
    PATIENCE= 10

    (X_train, y_train) , (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.2, random_state=11)

    print(X_train.shape)
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    validation_dataset = tf.data.Dataset.from_tensor_slices((X_validate, y_validate))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    train_dataset = train_dataset.map(lambda x, y: (tf.image.resize(x, [72,72], method=tf.image.ResizeMethod.BILINEAR), y), num_parallel_calls=AUTOTUNE)

    train_dataset = (
        train_dataset
        .cache()
        .shuffle(5 * BATCH_SIZE)
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
        dropout=0.25,
    )

    AdamW_optimizer= tfa.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer= AdamW_optimizer,
    metrics=["accuracy"]
    )
    

file_path= './saved_models/'
checkpoint = ModelCheckpoint(file_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
reduce_on_plateau = ReduceLROnPlateau(monitor='val_accuracy', mode="max", factor=0.33, patience=PATIENCE, verbose=1,min_lr=0.00002)
callbacks_list = [checkpoint, reduce_on_plateau]

model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks= callbacks_list,
)
model.save_weights(os.path.join(args.logdir, "vit"))
