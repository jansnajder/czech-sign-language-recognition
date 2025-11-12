import os
import numpy as np

from typing import Dict
from dataclasses import dataclass


@dataclass
class CoordinatesFile:
    file_name: str
    folder_name: str
    coordinates: np.array


def load_coordinates(folder: str) -> Dict[str, CoordinatesFile]:
    sub_folders = os.listdir(folder)

    data = {}

    for sub_folder in sub_folders:
        names = os.listdir(os.path.join(folder, sub_folder))

        for name in names:
            filepath = os.path.join(folder, sub_folder, name)
            data[name] = CoordinatesFile(name.strip(".csv"), sub_folder, np.genfromtxt(filepath, delimiter=','))

    return data


def load_coordinates_wo_subfolders(folder: str) -> Dict[str, np.array]:
    names = os.listdir(folder)
    data = {}

    for name in names:
        filepath = os.path.join(folder, name)
        data[name] = np.genfromtxt(filepath, delimiter=',')

    return data
