import tensorflow as tf
import numpy as np


# translate
def translate(inputs, transformer, tokenizer, max_length=138):
    if not isinstance(inputs, tf.Tensor):
        inputs = tf.constant(inputs, dtype=tf.float32)

    encoder_input = inputs[tf.newaxis]
    start = tokenizer.encode([tokenizer.BOS_TOKEN]) 
    end = tokenizer.encode([tokenizer.EOS_TOKEN])
    pad = tokenizer.encode([tokenizer.PAD_TOKEN])

    output = np.array([pad[0]] * max_length)[tf.newaxis]
    output[0][-1:] = start
    output_array = tf.constant(output, dtype=tf.int64)

    for _ in tf.range(1, max_length):
        predictions = transformer.predict((encoder_input, output_array), verbose=0)

        predictions = predictions[:, -1, :]  # Shape `(batch_size, 1, vocab_size)`.
        predicted_id = tf.argmax(predictions, axis=-1)
        output[0] = np.roll(output[0], -1)
        output[0][-1] = predicted_id[0]
        output_array = tf.constant(output, dtype=tf.int64)

        if predicted_id == end:
            break

    text = tokenizer.decode(output[0])  # Shape: `()`.

    return text