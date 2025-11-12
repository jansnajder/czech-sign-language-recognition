import csv
import os
import numpy as np

from typing import Dict, List
from dataclasses import dataclass

from .vocabulary import Vocabulary


@dataclass
class CoordinatesFile:
    file_name: str
    folder_name: str
    coordinates: np.array


# names
def load_names(path: str) -> List[str]:
    with open(path, 'r', encoding='utf-8') as fh:
        reader = csv.reader(fh)
        names = [line[0] for line in reader]

    return names


# coordinates
def load_coordinates(folder: str) -> Dict[str, np.array]:
    sub_folders = os.listdir(folder)
    data = {}

    for sub_folder in sub_folders:
        files = os.listdir(os.path.join(folder, sub_folder))

        for file in files:
            filepath = os.path.join(folder, sub_folder, file)
            data[file] = CoordinatesFile(file, sub_folder, np.genfromtxt(filepath, delimiter=','))

    return data


def pad_coordinates(data: Dict[str, np.array], pad_value: float = 0.0) -> Dict[str, np.array]:
    pad_length = max(item.coordinates.shape[0] for item in data.values())

    for name in data:
        data[name].coordinates = np.pad(data[name].coordinates, ((0, pad_length - data[name].coordinates.shape[0]), (0, 0)), mode='constant', constant_values=pad_value)

    return pad_length


# text
def load_text_data(path: str, vocab: Vocabulary) -> Dict[str, np.array]:
    text_data = {}

    with open(path, 'r', encoding='utf-8') as fh:
        reader = csv.reader(fh)
        text_data = {line[0].strip('.mp4'): vocab.encode(line[1:]) for line in reader}

    return text_data


def pad_text_data(data: Dict[str, np.array]) -> Dict[str, np.array]:
    to_pad = max([len(value) for value in data.values()]) + 1

    for name in data:
        data[name] = np.pad(data[name], (0, to_pad - len(data[name])), mode='constant', constant_values=Vocabulary.DEFAULT_PAD_ID)

    return to_pad
