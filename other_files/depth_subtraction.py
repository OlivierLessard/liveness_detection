import cv2
from PIL import Image
import time
import torch
import numpy as np
import os
import pandas as pd
import time
import datetime

import matplotlib.pyplot as plt


def get_time_str(start_time, curr_time):
    curr_time_str = datetime.datetime.fromtimestamp(curr_time).strftime('%Y-%m-%d %H:%M:%S')
    delta = datetime.timedelta(seconds=(curr_time - start_time))
    delta_str = str(delta - datetime.timedelta(microseconds=delta.microseconds))
    return curr_time_str, delta_str


def compute_depth_variance(img):
    flatten_values = img.flatten()
    return np.var(flatten_values)


fake_filenames = []
real_filenames = []
labels = []

# root = '/home/elham/anti-spoofing/Evaluation/benchmarks/input_images'
root_fake = 'depth/DATASET_VALIDATION/fake'
root_real = 'depth/DATASET_VALIDATION/real'
fakes = os.listdir(root_fake)
fakes = [x for x in fakes if '.db' not in x and x[-4:] == '.txt']
reals = os.listdir(root_real)
reals = [x for x in reals if '.db' not in x and x[-4:] == '.txt']
for fake in fakes:
    fake_filenames.append(root_fake + "/" + fake)
for real in reals:
    real_filenames.append(root_real + "/" + real)

for fake_filename in fake_filenames:
    file_number_fake = os.path.split(fake_filename)[1][:-6]

    for real_filename in real_filenames:
        file_number_real = os.path.split(real_filename)[1][:-6]
        if file_number_real == file_number_fake:
            fake_array = np.loadtxt(fake_filename, dtype=int)
            real_array = np.loadtxt(real_filename, dtype=int)
            fake_array_resized = np.resize(fake_array, (real_array.shape[0], real_array.shape[1]))
            real_array_resized = real_array
            difference = real_array_resized - fake_array_resized
            print("compute diff")

            scaled_array = (difference - difference.min()) / (difference.max() - difference.min()) * 255
            plt.imshow(scaled_array, cmap=plt.get_cmap('gray'))
            plt.savefig(fake_filename[:-4] + 'difference.jpg')
            plt.show(block=True)
            plt.close()

            # scaled_array = (real_array_resized - real_array_resized.min()) / (real_array_resized.max() - real_array_resized.min()) * 255
            # plt.imshow(scaled_array, cmap=plt.get_cmap('gray'))
            # plt.savefig(fake_filename[:-4] + 'real_array_resized.jpg')
            # plt.show(block=True)
            # plt.close()

