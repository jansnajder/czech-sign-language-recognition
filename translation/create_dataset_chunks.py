from __future__ import annotations
import json
import random
import tensorflow as tf

from data_handling.vocabulary import Vocabulary
from data_handling.data_handler import load_coordinates, pad_coordinates, load_names, load_text_data
from data_handling.data_handler import pad_text_data


def create_chunks(text_data, coordinates_data, folder, size: int = 1000):
    keys = list(coordinates_data.keys())
    random.shuffle(keys)
    coord_to_export = []
    text_to_export = []

    for idx, key in enumerate(keys):
        if idx != 0 and idx % size == 0:
            write_tfrecord(coord_to_export, text_to_export, f"{folder}/chunk_{idx//size - 1}.tfrecord")
            coord_to_export.clear()
            text_to_export.clear()

        coord_to_export.append(coordinates_data[key].coordinates)
        text_to_export.append(text_data[coordinates_data[key].folder_name])

    write_tfrecord(coord_to_export, text_to_export, f"{folder}/chunk_{idx//size}.tfrecord")


def write_tfrecord(inputs, outputs, filename):
    with tf.io.TFRecordWriter(filename) as writer:
        for idx in range(len(inputs)):
            example = serialize_example(inputs[idx], outputs[idx])
            writer.write(example)


def serialize_example(input_sequence, output_sequence):
    feature = {
        'input_sequence': tf.train.Feature(float_list=tf.train.FloatList(value=input_sequence.flatten())),
        'output_sequence': tf.train.Feature(int64_list=tf.train.Int64List(value=output_sequence.flatten()))
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


if __name__ == '__main__':
    # included names are only really small subset of original data
    train_names = load_names(f"names/train_names.csv")
    val_names = load_names(f"names/val_names.csv")
    test_names = load_names(f"names/test_names.csv")

    # vocab is created in create_vocab.py
    with open("vocab_full.json", "r", encoding="utf-8") as fh:
        data = json.load(fh)
        vocab = Vocabulary.from_dict(data)

    text_data = load_text_data("output_word_list.csv", vocab)
    text_pad = pad_text_data(text_data)

    # coordinates are not included
    coordinates_data = load_coordinates("inputs")
    coord_pad = pad_coordinates(coordinates_data)

    folder_to_file = {}

    for name, value in coordinates_data.items():
        if value.folder_name not in folder_to_file:
            folder_to_file[value.folder_name] = [name]
        else:
            folder_to_file[value.folder_name].append(name)

    train_data = {}
    val_data = {}
    test_data = {}
    all_names = train_names + val_names + test_names

    for name in all_names:
        if name in folder_to_file:
            if name in train_names:
                for file in folder_to_file[name]:
                    train_data[file] = coordinates_data.pop(file)
            elif name in val_names:
                for file in folder_to_file[name]:
                    if not "augmented" in file:
                        val_data[file] = coordinates_data.pop(file)
                        break
            elif name in test_names:
                for file in folder_to_file[name]:
                    if not "augmented" in file:
                        test_data[file] = coordinates_data.pop(file)
                        break

    create_chunks(text_data, train_data, f"train_chunks")
    create_chunks(text_data, val_data, f"val_chunks")
    create_chunks(text_data, test_data, f"test_chunks")
