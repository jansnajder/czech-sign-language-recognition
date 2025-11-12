import tensorflow as tf

from layers.encoder import Encoder
from layers.decoder import Decoder


class Transformer(tf.keras.Model):
    '''Transformer model, joining Encode and Decoder together, alsow allows to save the model.'''

    def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.vocab_size = vocab_size
        self.dout = dropout
        self.encoder = Encoder(
            num_layers=self.num_layers,
            d_model=self.d_model,
            num_heads=self.num_heads,
            dff=self.dff,
            dropout=self.dout
        )
        self.decoder = Decoder(
            num_layers=self.num_layers,
            d_model=self.d_model,
            num_heads=self.num_heads,
            dff=self.dff,
            vocab_size=self.vocab_size,
            dropout=self.dout
        )
        self.final_layer = tf.keras.layers.Dense(self.vocab_size)

    def call(self, inputs):
        context, x = inputs # .fit requires only one argumet
        context = self.encoder(context) # (batch_size, context_len, d_model)
        x = self.decoder(x, context) # (batch_size, target_len, d_model)
        logits = self.final_layer(x) # (batch_size, target_len, vocab_size)

        try:
            del logits._keras_mask
        except AttributeError:
            pass

        return logits

    def get_config(self):
        config = super().get_config()
        config['d_model'] = self.d_model
        config['num_layers'] = self.num_layers
        config['num_heads'] = self.num_heads
        config['dout'] = self.dout
        config['vocab_size'] = self.vocab_size
        config['dff'] = self.dff
        return config


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    '''Custom learning schedule, which starts with higher learning rate and rapidly decreases over the course of
    warmup_steps.
    '''
    def __init__(self, d_model, warmup_steps=18000):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        config = {}
        config['d_model'] = self.d_model.numpy()
        config['warmup_steps'] = self.warmup_steps
        return config


def masked_loss(label, pred):
    '''Loss function masking out the padding.'''
    mask = label != 0
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none'
    )
    loss = loss_object(label, pred)
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
    return loss


def masked_accuracy(label, pred):
    '''Accuracy function masking out the padding.'''
    pred = tf.argmax(pred, axis=2)
    label = tf.cast(label, pred.dtype)
    match = label == pred
    mask = label != 0
    match = match & mask
    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(match) / tf.reduce_sum(mask)