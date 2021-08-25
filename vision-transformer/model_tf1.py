import tensorflow.compat.v1 as tf
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    LayerNormalization,
)
#from tensorflow.keras.layers.experimental.preprocessing import Rescaling
tf.disable_v2_behavior()
tf.enable_eager_execution()

def Rescale(input, scale, offset=0):
    dtype = tf.float32
    scale = tf.cast(scale, dtype)
    offset = tf.cast(offset, dtype)
    return tf.cast(input, dtype) * scale + offset

def gelu(x):
    return 0.5 * x * (1.0 + tf.math.erf(x / tf.cast(tf.sqrt(2.0), x.dtype)))

class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = Dense(embed_dim)
        self.key_dense = Dense(embed_dim)
        self.value_dense = Dense(embed_dim)
        self.combine_heads = Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))
        output = self.combine_heads(concat_attention)
        return output


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, mlp_dim_l2, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.mlp = tf.keras.Sequential(
            [   Dense(mlp_dim_l2, activation=gelu ),
                Dropout(dropout),
                Dense(embed_dim),
                Dropout(dropout), 
            ]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def call(self, inputs, training):
        inputs_norm = self.layernorm1(inputs)
        attn_output = self.att(inputs_norm)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = attn_output + inputs

        out1_norm = self.layernorm2(out1)
        mlp_output = self.mlp(out1_norm)
        mlp_output = self.dropout2(mlp_output, training=training)
        return mlp_output + out1


class VisionTransformer(tf.keras.Model):
    def __init__(
        self,
        image_size,
        patch_size,
        num_layers,
        num_classes,
        embedding_dim,
        num_heads,
        mlp_dim_l1,
        mlp_dim_l2,
        channels=3,
        dropout=0.1,
    ):
        super(VisionTransformer, self).__init__()
        num_patches = (image_size // patch_size) ** 2
        #72/6 **2 = 144

        #3*6*6= 108
        self.patch_dim = channels * (patch_size ** 2)

        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        #Adding learnable Positional Encoding embedding weights to the model class
        self.pos_emb = self.add_weight( "pos_emb", shape=(1, num_patches + 1, embedding_dim))

        #Adding learnable classification embedding weights to the model class
        self.class_emb = self.add_weight("class_emb", shape=(1, 1, embedding_dim))
        self.patch_proj = Dense(embedding_dim)
        self.enc_layers = [TransformerBlock(embedding_dim, num_heads, mlp_dim_l2, dropout) for _ in range(num_layers)]
        self.mlp_head = tf.keras.Sequential(
            [
                LayerNormalization(epsilon=1e-6),
                Dense(mlp_dim_l1, activation=gelu),
                Dropout(dropout),
                Dense(mlp_dim_l2, activation='relu'),
                Dense(num_classes),
            ]
        )

    def extract_patches(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patches = tf.reshape(patches, [batch_size, -1, self.patch_dim])
        return patches

    def call(self, x, training):
        batch_size = tf.shape(x)[0]
        x = Rescale(x, 1.0 / 255.0)

        patches = self.extract_patches(x)
        x = self.patch_proj(patches)
        classification_emb = tf.broadcast_to( self.class_emb, [batch_size, 1, self.embedding_dim])
        x = tf.concat([classification_emb, x], axis=1)
        x = x + self.pos_emb

        for layer in self.enc_layers:
            x = layer(x, training)

        # First (class token) is used for classification
        x = self.mlp_head(x[:, 0])
        return x
