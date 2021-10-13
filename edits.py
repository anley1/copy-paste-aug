import cv2
import numpy as np
import json
from copy_paste import CopyPaste
from coco import CocoDetectionCP
from visualize import display_instances
import albumentations as A
import random
from matplotlib import pyplot as plt


def edit_unique_anns():
    with open("data/aug.json", 'r') as j_file:
        data = json.load(j_file)

    anno_len = len(data['annotations'])

    count = 0
    for ix in range(anno_len):
        data['annotations'][ix]['id'] = count
        count += 1

    with open("new.json", 'w') as j_file:
        json.dump(data, j_file)


def edit_jpg_ext():
    with open("data/aug.json", 'r') as j_file:
        data = json.load(j_file)

    img_len = len(data['images'])

    for ix in range(img_len):
        data['images'][ix]['file_name'] = data['images'][ix]['file_name'] + \
                                          '.jpg'

    with open("new.json", 'w') as j_file:
        json.dump(data, j_file)


if __name__ == "__main__":
    edit_jpg_ext()