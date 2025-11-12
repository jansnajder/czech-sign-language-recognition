import tensorflow as tf


class FeedForward(tf.keras.layers.Layer):

    def __init__(self, d_model, dff, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dff = dff
        self.dout = dropout
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(self.dff, activation='relu'),
            tf.keras.layers.Dense(self.d_model),
            tf.keras.layers.Dropout(self.dout)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x

    def get_config(self):
        config = super().get_config()
        config['d_model'] = self.d_model
        config['dout'] = self.dout
        config['dff'] = self.dff
        return config