# Vision_Transformers_tf1


# Requirements:

tensorflow 1.15  :If using CUDA, recommended approach is to install tf2, followed by import tensorflow.compat.v1 as tf, tf.disable_v2_behavior() and tf.enable_eager_execution()
sklearn
numpy


# To train on Cifar 10:

python train_tf1_cifar10_aug.py


# To train on Cifar 100:
python train_tf1_cifar100_aug.py



# Recommended Hyper-parameters:

    IMAGE_SIZE= 32
    NUMBER_OF_CLASSES= 10
    PATCH_SIZE= 4
    NUMBER_OF_LAYERS=8
    EMBEDDING_DIM=64
    NUM_HEADS= 4
    MLP_DIM_L1= 32
    MLP_DIM_L2= 128
    LEARNING_RATE= 0.001 
    BATCH_SIZE= 256
    EPOCHS= 100#300
    PATIENCE= 6
    
    
   
  You can also change the Learning Rate reduction strategy by modifying _factor_ r in ReduceLROnPlateau(monitor="val_Top-1-accuracy", mode="max", factor=0.5, patience=PATIENCE, verbose=1,min_lr=0.00002) in the train-....py files as well as the _PATIENCE_ hyper-parameter.

