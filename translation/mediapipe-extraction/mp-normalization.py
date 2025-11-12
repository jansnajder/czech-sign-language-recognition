import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from data_handlers.coordinates import load_coordinates


if __name__ == '__main__':
    '''Normalization of the data, the MinMaxScaler must be expored to be used during inference.'''
    folder = ''
    output = 'output'
    coordinates_dict = load_coordinates(folder)

    min_ = 0
    max_ = 0
    i = 0

    scaler = MinMaxScaler((-1, 1))

    for name, coordinate_file in coordinates_dict.items():
        coordinates_dict[name].coordinates = coordinate_file.coordinates[~np.all(coordinate_file.coordinates==0, axis=1)]
        scaler.fit(coordinates_dict[name].coordinates)

    for name, coordinate_file in coordinates_dict.items():
        normalized = scaler.transform(coordinate_file.coordinates)
        os.makedirs(os.path.join(output, coordinate_file.folder_name), exist_ok=True)
        np.savetxt(os.path.join(output, coordinate_file.folder_name, f"{coordinate_file.file_name}.csv"), normalized, delimiter=",")
