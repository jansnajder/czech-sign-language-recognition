import tensorflow as tf

from .attentions import CausalSelfAttention, CrossAttention
from .feed_forward import FeedForward
from .embeddings import TokenAndPositionalEmbedding


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dout = dropout
        self.causual_self_attention = CausalSelfAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model,
            dropout=self.dout
        )
        self.cross_attention = CrossAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model,
            dropout=self.dout
        )
        self.ffn = FeedForward(self.d_model, self.dff)
        self.last_attn_scores = None

    def call(self, x, context):
        x = self.causual_self_attention(x)
        x = self.cross_attention(x, context)

        self.last_attn_scores = self.cross_attention.last_attn_scores
        x  = self.ffn(x)
        return x

    def get_config(self):
        config = super().get_config()
        config['d_model'] = self.d_model
        config['num_heads'] = self.num_heads
        config['dout'] = self.dout
        config['dff'] = self.dff
        return config


class Decoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size, dropout=0.1):
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.vocab_size = vocab_size
        self.dout = dropout

        self.embedding = TokenAndPositionalEmbedding(
            vocab_size=self.vocab_size,
            d_model=self.d_model
        )
        self.dropout = tf.keras.layers.Dropout(self.dout)
        self.dec_layers = [
            DecoderLayer(
                d_model=self.d_model,
                num_heads=self.num_heads,
                dff=self.dff,
                dropout=self.dout
            ) for _ in range(self.num_layers)
        ]

        self.last_attn_scores = None

    def call(self, x, context):
        # x is output sentence shape: (batch, target_seq_len)
        x = self.embedding(x) # (batch, targetr_seq_len, d_model)
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, context)

        self.last_attn_scores = self.dec_layers[-1].last_attn_scores
        return x

    def get_config(self):
        config = super().get_config()
        config['d_model'] = self.d_model
        config['num_layers'] = self.num_layers
        config['num_heads'] = self.num_heads
        config['dout'] = self.dout
        config['vocab_size'] = self.vocab_size
        config['dff'] = self.dff
        return config