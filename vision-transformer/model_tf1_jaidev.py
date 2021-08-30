import tensorflow.compat.v1 as tf
tf.enable_eager_execution()
from MHA import MultiHeadSelfAttention

from tensorflow.keras.layers import (
    Dense,
    Dropout,
    LayerNormalization,
)

def Rescale(input, scale, offset=0):
    """Rescaling helper function to scale image elements down to the range [0,1]"""
    dtype = tf.float32
    scale = tf.cast(scale, dtype)
    offset = tf.cast(offset, dtype)
    return tf.cast(input, dtype) * scale + offset

def gelu(x):
    """ The GELU Activation function: defined as x*CDF(x) for the Standard Normal(0,1) Distribution"""

    return 0.5 * x * (1.0 + tf.math.erf(x / tf.cast(tf.sqrt(2.0), x.dtype)))


class MLP():
    def __init__(self, layer_1, layer_2, rate=0.2):
        tf.keras.Sequential(
            [   Dense(layer_1, activation=gelu ),
                Dropout(rate),
                Dense(layer_2,  activation=gelu),
            ]
        )

class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, mlp_l2_dim):
        super(TransformerEncoder, self).__init__()
        self.mlp = MLP(mlp_l2_dim, embed_dim,0.2)
        self.MHA_layer = MultiHeadSelfAttention(embed_dim, num_heads)
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(0.2)
        self.dropout2 = Dropout(0.2)

    def call(self, input_embeddings, training=True):
        output = self.att(input_embeddings)
        output = self.layernorm1(output)
        
        output = self.dropout1(output, training=training)
        output_1 = output + input_embeddings
        #Skip Connection: Adding input_embeddings to the output 

        output_norm = self.layernorm2(output_1)
        MLP_output = self.mlp(output)
        MLP_output = self.dropout2(MLP_output, training=training)
        return MLP_output + output_1 
        #Skip Connection: Adding output_1 to the final output MLP_output 

class PatchExtractEncoder(tf.keras.layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchExtractEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = tf.keras.layers.Dense(units=projection_dim)
        self.position_embedding = tf.keras.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def get_patches(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

    def call(self, images):
        patches = self.get_patches(images)
        positions = tf.range(start=0, limit=self.num_patches, delta=1)

        #Adding learnable Positional Encoding embedding weights to the model class. A positional embedding learns to represent the position of each specific patch number of the image
        proj_patches = self.projection(patches) + self.position_embedding(positions)
        return proj_patches

class VisionTransformer(tf.keras.Model):
    def __init__(
        self,
        image_size,
        patch_size,
        patch_stride,
        num_layers,
        num_classes,
        embedding_dim,
        num_heads,
        mlp_dim_l1,
        mlp_dim_l2,
        dropout=0.1,
    ):
        super(VisionTransformer, self).__init__()
        num_patches = (image_size // patch_size) ** 2
        #The number of patches is analagous to the number of words in a sequence being fed to a transformer. The image patches are flattened and transformed to a lower dimensional embedding space (embedding dim) 
        self.patch_dim = (patch_size ** 2) * 3
        #Flatting the path results in a path_dim dimensional vector. For patch_size =4, this is 3*4*4 = 48 dimensional
        self.patch_size = patch_size
        self.patch_stride= patch_stride
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.PatchExtractEncoder= PatchExtractEncoder(num_patches,embedding_dim)
        self.transformer_layers = [TransformerEncoder(embedding_dim, num_heads, mlp_dim_l2, dropout) for _ in range(num_layers)]
        self.layernorm= tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.classifier = tf.keras.Sequential(
            [
                Dense(mlp_dim_l1, activation=gelu),
                Dropout(0.3),
                Dense(mlp_dim_l2, activation=gelu),
                Dropout(0.2),
                Dense(num_classes, activation='relu'),
            ]
        )
        self.flatten= tf.keras.layers.Flatten()


    def call(self, images, training=True):
        batch_size = tf.shape(images)[0]
        images = Rescale(images, 1.0 / 255.0)
        #Image elements are scaled by 1/255 so that each element of the image x is now between 0 and 1 

         #Extract patches using specified patch_size and patch_stride parameters, and return flatten patches of shape [batch_size, number of patches, self.patch_dim], Flattened Patches are projected down to (embedding_dim) sized embeddings
        proj_patches = self.PatchExtractEncoder(images)

        for transformer_encoder in self.transformer_layers:
            x = transformer_encoder(x, training)

        x = self.layernorm(x)
        x= self.flatten(x)
        x = self.classifier(x)
        #We use the first token from the outputs of the last transformer block as this is the Classification token
        return x
