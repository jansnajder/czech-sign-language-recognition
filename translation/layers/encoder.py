import tensorflow as tf

from .attentions import GlobalSelfAttention
from .feed_forward import FeedForward
from .embeddings import PositionalEmbedding


class EncodeLayer(tf.keras.layers.Layer):

    def __init__(self, *, d_model, num_heads, dff, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout = dropout
        self.self_attention = GlobalSelfAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model,
            dropout=self.dropout
        )
        self.ffn = FeedForward(self.d_model, self.dff)

    def call(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x

    def get_config(self):
        config = super().get_config()
        config['d_model'] = self.d_model
        config['num_heads'] = self.num_heads
        config['dropout'] = self.dropout
        config['dff'] = self.dff
        return config

class Encoder(tf.keras.layers.Layer):

    def __init__(self, *, num_layers, d_model, num_heads, dff, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dout = dropout
        self.num_layers = num_layers
        self.embed_custom = tf.keras.Sequential([
            tf.keras.layers.Dense(self.d_model),
            tf.keras.layers.Dropout(self.dout)
        ])
        self.pos_embedding = PositionalEmbedding(self.d_model)
        self.enc_layers = [
            EncodeLayer(
                d_model=self.d_model,
                num_heads=self.num_heads,
                dff=self.dff,
                dropout=self.dout
            ) for _ in range(self.num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(self.dout)

    def call(self, x):
        # x is mediapipe output (batch, seq_len, 100)
        x = self.embed_custom(x)
        x = self.pos_embedding(x)
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x)

        return x

    def get_config(self):
        config = super().get_config()
        config['num_layers'] = self.num_heads
        config['d_model'] = self.d_model
        config['num_heads'] = self.num_heads
        config['dout'] = self.dout
        config['dff'] = self.dff
        return config