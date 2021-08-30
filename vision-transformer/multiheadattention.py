import tensorflow.compat.v1 as tf
tf.enable_eager_execution()


from tensorflow.keras.layers import (
    Dense,
    Dropout,
    LayerNormalization,
)


class MultiHeadSelfAttention(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert (
            embed_dim % num_heads == 0
        ), "embedding dimension not divisible by num heads"
        self.projection_dim = embed_dim // num_heads
        self.wq = tf.keras.layers.Dense(embed_dim)
        self.wk = tf.keras.layers.Dense(embed_dim)
        self.wv = tf.keras.layers.Dense(embed_dim)
        self.combine_heads = tf.keras.layers.Dense(embed_dim)

    def attention(self, q, k, v):
        score = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dk)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, v)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, x):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(x)[0]
        q = self.wq(x)  # (batch_size, seq_len, embed_dim)
        k = self.wk(x)  # (batch_size, seq_len, embed_dim)
        v = self.wv(x)  # (batch_size, seq_len, embed_dim)
        q = self.separate_heads(
            q, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        k = self.separate_heads(
            k, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        v = self.separate_heads(
            v, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(q, k, v)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return output