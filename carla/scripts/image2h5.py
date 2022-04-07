import cv2
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import pathlib
import h5py
import glob
import argparse
from os import listdir


def image2h5(args):
    images = read_images(args.image_dir, args.start_idx, args.num_images)
    store_many_hdf5(images, args.output_file)


def read_images(data_dir, start_idx, num_images):
    image_names = sorted(listdir(data_dir))[start_idx: start_idx + num_images]
    images = np.vstack([np.expand_dims(cv2.imread(os.path.join(data_dir, image)), axis=0) for image in tqdm(image_names)])

    for i in range(images.shape[0]):
        images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
    return images


def store_many_hdf5(images,  filename):
    # Create a new HDF5 file
    with h5py.File(filename, "w") as file:
        # Create a dataset in the file
        file.create_dataset(
            "images", np.shape(images), h5py.h5t.STD_U8BE, data=images, compression='gzip', chunks=(1, 320, 480, 3)
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--num_images', type=int)
    parser.add_argument('--output_file', type=str, required=True)
    args = parser.parse_args()
    image2h5(args)

