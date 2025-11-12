import os
import numpy as np

from data_handlers.coordinates import load_coordinates_wo_subfolders


if __name__ == '__main__':
    '''Script that strips frames where MPP did not find the upper body'''
    folder = ''
    output = 'output'
    coordinates_dict = load_coordinates_wo_subfolders(folder)

    for name, value in coordinates_dict.items():
        coordinates_dict[name] = value[~np.all(value==0, axis=1)]
        np.savetxt(os.path.join(output, name), coordinates_dict[name], delimiter=",")
