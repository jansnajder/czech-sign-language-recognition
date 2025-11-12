import numpy as np
import os

from data_handlers.coordinates import load_coordinates_wo_subfolders


def scale(coordinates, range_=(0.8, 1.2)):
    factor = np.random.uniform(*range_)
    coordinates *= factor
    return coordinates


def rotate(coordinates, range_=(-15, 15)):
    angle = np.radians(np.random.uniform(*range_))
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    coords_reshaped  = coordinates.reshape(-1, 2)
    rotated_coords = np.dot(coords_reshaped, rotation_matrix)
    rotated_coord_reshaped = rotated_coords.reshape(coordinates.shape)
    return rotated_coord_reshaped


def shear(coordinates, range_x=(-0.15, 0.15), range_y=(-0.15, 0.15)):
    shear_x = np.random.uniform(*range_x)
    shear_y = np.random.uniform(*range_y)
    shear_matrix = np.array([[1, shear_x], [shear_y, 1]])
    coords_reshaped  = coordinates.reshape(-1, 2)
    sheared_coords = np.dot(coords_reshaped, shear_matrix)
    sheared_coords_reshaped = sheared_coords.reshape(coordinates.shape)
    return sheared_coords_reshaped


def jitter(coordinates, std_devs, noise_std=0.01):
    noise = np.array([[np.random.normal(0, std_dev*noise_std) for std_dev in std_devs] for _ in range(coordinates.shape[0])])
    coordinates += noise
    return coordinates


if __name__ == '__main__':
    JITTER = 0.5
    SCALE = 0.5
    ROTATION = 0.5
    SHEAR = 0.25

    input_folder = ''
    coordinates_dict = load_coordinates_wo_subfolders(input_folder)
    output_folder = "output"

    for name, coordinates in coordinates_dict.items():
        std_devs = np.std(coordinates, axis=0)
        name_stripped = name.strip(".csv")
        coord_folder = os.path.join(output_folder, name_stripped)
        os.makedirs(coord_folder, exist_ok=True)

        for i in range(3):
            augmented_coords = coordinates.copy()
            changed = 0

            while not changed:
                # jitter
                if np.random.rand() < JITTER:
                    augmented_coords = jitter(augmented_coords, std_devs)
                    changed += 1

                # scale
                if np.random.rand() < SCALE:
                    augmented_coords = scale(augmented_coords)
                    changed += 1

                # rotation
                if np.random.rand() < ROTATION:
                    augmented_coords = rotate(augmented_coords)
                    changed += 1

                # shear
                if np.random.rand() < SHEAR:
                    augmented_coords = shear(augmented_coords)
                    changed += 1

            np.savetxt(os.path.join(coord_folder, f"{name_stripped}_augmented_{i}.csv"), augmented_coords, delimiter=",")