# -*- coding: utf-8 -*-
"""new_v0.0.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1gsGjQxHvdjTF-6Rr6R5VQC11T5YkhQLa
"""

#!pip install efficientnet_pytorch

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2

from tqdm import tqdm_notebook as tqdm

import random
import time
import sys
import os
import math

import shutil



def split_dataset_into_3(path_to_dataset, train_ratio, valid_ratio):
    """
    wrapper function to split the dataset into three subsets(test,validation,train)
    :param path_to_dataset:
    :param train_ratio:
    :param valid_ratio:
    :return:
    """
    _, sub_dirs, _ = next(iter(os.walk(path_to_dataset)))  # retrieve name of subdirectories
    sub_dir_item_cnt = [0 for i in range(len(sub_dirs))]  # list for counting items in each sub directory(class)

    # directories where the splitted dataset will lie
    dir_train = os.path.join(os.path.join(os.path.dirname(path_to_dataset), 'DATASET'), 'train')
    dir_valid = os.path.join(os.path.join(os.path.dirname(path_to_dataset), 'DATASET'), 'validation')
    dir_test = os.path.join(os.path.join(os.path.dirname(path_to_dataset), 'DATASET'), 'test')

    for i, sub_dir in enumerate(sub_dirs):

        dir_train_dst = os.path.join(dir_train, sub_dir)  # directory for destination of train dataset
        dir_valid_dst = os.path.join(dir_valid, sub_dir)  # directory for destination of validation dataset
        dir_test_dst = os.path.join(dir_test, sub_dir)  # directory for destination of test dataset

        print(dir_train_dst)
        print(dir_valid_dst)
        print(dir_test_dst)

        # variables to save the sub directory name(class name) and to count the images of each sub directory(class)
        class_name = sub_dir
        sub_dir = os.path.join(path_to_dataset, sub_dir)
        sub_dir_item_cnt[i] = len(os.listdir(sub_dir))
        print(sub_dir)
        print(sub_dir_item_cnt[i])

        items = os.listdir(sub_dir)

        # transfer data to trainset
        for item_idx in range(round(sub_dir_item_cnt[i] * train_ratio)):
            if not os.path.exists(dir_train_dst):
                os.makedirs(dir_train_dst)

            source_file = os.path.join(sub_dir, items[item_idx])
            dst_file = os.path.join(dir_train_dst, items[item_idx])
            shutil.copyfile(source_file, dst_file)

        # transfer data to validation
        for item_idx in range(round(sub_dir_item_cnt[i] * train_ratio) + 1,
                              round(sub_dir_item_cnt[i] * (train_ratio + valid_ratio))):
            if not os.path.exists(dir_valid_dst):
                os.makedirs(dir_valid_dst)

            source_file = os.path.join(sub_dir, items[item_idx])
            dst_file = os.path.join(dir_valid_dst, items[item_idx])
            shutil.copyfile(source_file, dst_file)

        # transfer data to testset
        for item_idx in range(round(sub_dir_item_cnt[i] * (train_ratio + valid_ratio)) + 1, sub_dir_item_cnt[i]):
            if not os.path.exists(dir_test_dst):
                os.makedirs(dir_test_dst)

            source_file = os.path.join(sub_dir, items[item_idx])
            dst_file = os.path.join(dir_test_dst, items[item_idx])
            shutil.copyfile(source_file, dst_file)

    return


#split_dataset_into_3('./datasets', 0.9, 0.1)  # only generating train and validation sets


def annotate(BASE_PATH):
    image = []
    labels = []
    encoded_labels = []
    for file in os.listdir(BASE_PATH):
        if file == 'infected':
            for c in os.listdir(os.path.join(BASE_PATH, file)):
                if c != 'annotations':
                    image.append(c)
                    labels.append(0)
                    encoded_labels.append('infected')
        if file == 'healthy':
            for c in os.listdir(os.path.join(BASE_PATH, file)):
                if c != 'annotations':
                    image.append(c)
                    labels.append(1)
                    encoded_labels.append('healthy')
    data = {'Images': image, 'Labels': labels, 'Encoded_labels': encoded_labels}
    data = pd.DataFrame(data)
    return data

