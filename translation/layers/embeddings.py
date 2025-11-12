import numpy as np
import tensorflow as tf


def positional_encoding(length, depth):
    depth = depth / 2

    positions = np.arange(length)[:, np.newaxis] # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth # (1, depth)

    angle_rates = 1 / (10000**depths)   # (1, depth)
    angle_rads = positions * angle_rates # (pos, depth)

    pos_encoding = np.concatenate(
        [np.sin(angle_rads) , np.cos(angle_rads)],
        axis=-1
    )

    return tf.cast(pos_encoding, dtype=tf.float32)


class TokenAndPositionalEmbedding(tf.keras.layers.Layer):

    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.d_model, mask_zero=True)
        self.pos_encoding = positional_encoding(length = 4096, depth=self.d_model)

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x

    def get_config(self):
        config = super().get_config()
        config['vocab_size'] = self.vocab_size
        config['d_model'] = self.d_model
        return config


class PositionalEmbedding(tf.keras.layers.Layer):

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.pos_encoding = positional_encoding(length=4096, depth=self.d_model)

    def call(self, x):
        length = tf.shape(x)[1]
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x

    def get_config(self):
        config = super().get_config()
        config['d_model'] = self.d_model
        return config