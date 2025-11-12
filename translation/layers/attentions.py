import tensorflow as tf


class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, *, num_heads, key_dim, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dout = dropout
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.key_dim, dropout=self.dout)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()
        self.last_attn_scores = None

    def get_config(self):
        config = super().get_config()
        config['num_heads'] = self.num_heads
        config['key_dim'] = self.key_dim
        config['dout'] = self.dout
        return config


class CrossAttention(BaseAttention):

    def call(self, x, context):
        attn_output, attn_scores = self.mha(
            query=x,
            key=context,
            value=context,
            return_attention_scores=True
        )

        self.last_attn_scores = attn_scores
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class GlobalSelfAttention(BaseAttention):

    def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x
        )
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class CausalSelfAttention(BaseAttention):

    def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x,
            use_causal_mask=True
        )
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x