import tensorflow.compat.v1 as tf
from MHA import MultiHeadSelfAttention
tf.enable_eager_execution()


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

def _gelu_smooth(x):
    x = tf.convert_to_tensor(x)
    pi = tf.cast(math.pi, x.dtype)
    coeff = tf.cast(0.044715, x.dtype)
    cdf = 0.5 * (1.0 + tf.tanh(tf.sqrt(2.0 / pi) * (x + coeff * tf.pow(x, 3))))

    return x * cdf


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

class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, mlp_dim_l2, dropout=0.1):
        super(TransformerEncoder, self).__init__()
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
        #Skip Connection: Adding inputs to the attn_output 

        out1_norm = self.layernorm2(out1)
        mlp_output = self.mlp(out1_norm)
        mlp_output = self.dropout2(mlp_output, training=training)
        return mlp_output + out1 
        #Skip Connection: Adding out1 to the final output mlp_output 


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
        channels=3,
        dropout=0.1,
    ):
        super(VisionTransformer, self).__init__()
        num_patches = (image_size // patch_size) ** 2
        #The number of patches is analagous to the number of words in a sequence being fed to a transformer. The image patches are flattened and transformed to a lower dimensional embedding space (embedding dim) 
        self.patch_dim = channels * (patch_size ** 2)
        #Flatting the path results in a path_dim dimensional vector. For patch_size =4, this is 3*4*4 = 48 dimensional

        self.patch_size = patch_size
        self.patch_stride= patch_stride
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        #Adding learnable Positional Encoding embedding weights to the model class. A positional embedding learns to represent the position of each specific patch number of the image
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
                Dense(mlp_dim_l2, activation=gelu),
                Dense(num_classes),
            ]
        )

    def extract_patches(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_stride, self.patch_stride, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patches = tf.reshape(patches, [batch_size, -1, self.patch_dim])
        return patches

    def call(self, x, training):
        batch_size = tf.shape(x)[0]
        x = Rescale(x, 1.0 / 255.0)
        #Image elements are scaled by 1/255 so that each element of the image x is now between 0 and 1 

        patches = self.extract_patches(x)
        #Extract patches using specified patch_size and patch_stride parameters, and return flatten patches of shape [batch_size, number of patches, self.patch_dim]

        x = self.patch_proj(patches)
        #Flattened Patches are projected down to (embedding_dim) sized embeddings

        classification_emb = tf.broadcast_to( self.class_emb, [batch_size, 1, self.embedding_dim])
        x = tf.concat([classification_emb, x], axis=1)
        #Concatenating the classification embedding to the patch_embeddings such that x now has (patches+1) embeddings along axis 1

        x = x + self.pos_emb
        # the positional embeddings are added to the patch embeddings. The patch embeddings within now also have positional information encoded

        for layer in self.enc_layers:
            x = layer(x, training)

        x = self.mlp_head(x[:, 0])
        #We use the first token from the outputs of the last transformer block as this is the Classification token
        return x
