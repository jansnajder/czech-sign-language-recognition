import os
import numpy as np

from data_handlers.coordinates import load_coordinates


if __name__ == '__main__':
    '''Downsampling of coordinates.'''
    input_ = ''
    coordinates_dict = load_coordinates(input_)

    DOWNSAMPLING = 3

    for name, coordinates in coordinates_dict.items():
        folder = os.path.join("output", coordinates.folder_name)
        os.makedirs(folder, exist_ok=True)

        for i in range(0, DOWNSAMPLING):
            downsampled = coordinates.coordinates[i::DOWNSAMPLING]
            np.savetxt(os.path.join(folder, f"{coordinates.file_name}_{i}.csv"), downsampled, delimiter=",")