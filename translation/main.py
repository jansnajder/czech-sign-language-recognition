import os
import json
import random
import tensorflow as tf

from data_handling.vocabulary import Vocabulary
from transformer import Transformer, CustomSchedule, masked_loss, masked_accuracy
from evaluation.translate import translate
from evaluation.metrics import calculate_bleu, calculate_rouge


def parse_example(example_proto):
    '''Parse items from chunks. Chunks were used to allow GPU fetching and to speed up the training process.'''
    # Define your feature description dictionary
    feature_description = {
        'input_sequence': tf.io.FixedLenFeature([539 * 400], tf.float32),  # length * coordinates
        'output_sequence': tf.io.FixedLenFeature([139], tf.int64)  # max output length
    }
    # Parse the input tf.train.Example proto
    example = tf.io.parse_single_example(example_proto, feature_description)
    input_sequence = tf.reshape(example['input_sequence'], [539, 400])  # Reshape the input sequence
    output_sequence = tf.reshape(example['output_sequence'], [139])  # Reshape the output sequence
    return (input_sequence, output_sequence[:-1]), output_sequence[1:]


if __name__ == '__main__':
    batch_size = 8
    num_layers = 4
    d_model = 1024
    dff = 1024
    num_heads = 4
    dropout = 0.2
    model_name = f'model'
    # chunks are created in create_dataset_chunks.py, however each chunk was nearly 1 GB, so they are not included
    train_chunks_folder = "train_chunks"
    val_chunks_folder = "val_chunks"
    train_chunks = os.listdir(train_chunks_folder)
    val_chunks = os.listdir(val_chunks_folder)

    # vocab is created in create_vocab.py
    with open("vocab_full.json", "r", encoding="utf-8") as fh:
        data = json.load(fh)
        vocab = Vocabulary.from_dict(data)

    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    with strategy.scope():
        transformer = Transformer(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            vocab_size=vocab.size,
            dropout=dropout
        )
        lr = CustomSchedule(d_model)
        opt = tf.keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        transformer.compile(
            loss=masked_loss,
            optimizer=opt,
            metrics=[masked_accuracy],
        )

    try:
        with open(f"{model_name}.txt", "w", encoding="utf-8") as fh:
            for i in range(500):
                random.shuffle(train_chunks)

                for _ in range(20):
                    train_chunk = random.choice(train_chunks)
                    val_idx = random.randint(0, len(val_chunks) - 1)
                    train_chunk_path = os.path.join(train_chunks_folder, train_chunk)
                    val_chunk_path = os.path.join(val_chunks_folder, val_chunks[val_idx])

                    train_dataset = tf.data.TFRecordDataset(train_chunk_path).map(parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
                    val_dataset = tf.data.TFRecordDataset(val_chunk_path).map(parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)

                    train_batch = train_dataset.shuffle(buffer_size=1000).batch(batch_size, drop_remainder=True).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
                    val_batch = val_dataset.shuffle(buffer_size=1000).batch(batch_size, drop_remainder=True)

                    transformer.fit(
                        train_batch,
                        validation_data=val_batch
                    )

                print(f"Saving model at epoch {i}")
                transformer.save(f"output_models/{model_name}_{i}.tf", save_format="tf", overwrite=True, save_traces=True)

                # Each epoch there are 10 predictions showed as well as ROUGE and BLEU calculated to better describe
                # the learning of the model.
                fh.write(f"\n\nEpoch #{i}:")
                val_list = (val_dataset.take(10).as_numpy_iterator())
                ground_truths = []
                predictions = []

                for sample in val_list:
                    prediction = translate(sample[0][0], transformer, vocab, max_length=sample[0][1].shape[0])
                    gt = ' '.join(vocab.decode(sample[0][1]))
                    pred = ' '.join(prediction)
                    fh.write(f"\nPrediction: {pred}")
                    fh.write(f"\nGround truth: {gt}")

                    print(f"Prediction: {pred}")
                    print(f"Ground truth: {gt}")

                    predictions.append(pred)
                    ground_truths.append(gt)

                rouge_f = calculate_rouge(ground_truths, predictions)
                print(f"rougeL: {rouge_f}")
                fh.write(f"\nrougeL: {rouge_f}")

                bleus = calculate_bleu(ground_truths, predictions)
                print(f"bleu1: {bleus[0]}, bleu2: {bleus[1]}, bleu3: {bleus[2]}, bleu4: {bleus[3]}")
                fh.write(f"\nbleu1: {bleus[0]}, bleu2: {bleus[1]}, bleu3: {bleus[2]}, bleu4: {bleus[3]}")


    except KeyboardInterrupt:
        pass


    transformer.save(f'output_models/{model_name}.tf', save_format='tf', overwrite=True, save_traces=True)
