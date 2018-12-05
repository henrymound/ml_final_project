import cv2
import csv
from PIL import Image
import tensorflow as tf
from random import shuffle
import sys
import numpy as np
import glob
import json
from os import listdir
from math import ceil

GLOBAL_IMAGE_SIZE = 50
IMAGES_TO_PARSE = 10000000
PREFIX_PATH = "classifier/signDatabasePublicFramesOnly/"
LABELS_TRAIN_PATH = "bdd100k_labels/labels/bdd100k_labels_images_train.json"
IMAGES_TRAIN_PATH = 'bdd100k_images/images/100k/train/'
LABELS_VALIDATION_PATH = "bdd100k_labels/labels/bdd100k_labels_images_val.json"
IMAGES_VALIDATION_PATH = 'bdd100k_images/images/100k/val/'

# Used to convert images and annotations to TensorFlow compatible bytestring
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def extract_data():
    print("\n****Berkeley Deep Drive Data Extraction****\n")
    print("Using global image size of %i x %i pixels" %(GLOBAL_IMAGE_SIZE, GLOBAL_IMAGE_SIZE) )

    print("**Gathering labels for training set:**")
    labels_dict = get_labels(LABELS_TRAIN_PATH)
    print("%i labels found" %len(labels_dict))
    print("*Collecting image filenames*")
    train_images = get_image_filenames(IMAGES_TRAIN_PATH)
    print("*Formatting and encoding images and labels*")
    prep_image_set('train.tfrecords', train_images, IMAGES_TRAIN_PATH, labels_dict)
    print("*Training data extracted*\n")

    print("**Gathering labels for validation set:**")
    validation_labels_dict = get_labels(LABELS_VALIDATION_PATH)
    print("%i labels found" %len(validation_labels_dict))
    print("*Collecting image filenames*")
    validation_images = get_image_filenames(IMAGES_VALIDATION_PATH)
    print("*Formatting and encoding images and labels*")
    prep_image_set('validation.tfrecords', validation_images,
        IMAGES_VALIDATION_PATH, validation_labels_dict)
    print("*Validation data extracted*")

    print("\n*** Data Extraction Complete ***")
    print("Use train.tfrecords and validation.tfrecords")

def prep_image_set(set_name, images, images_path, labels):
    image_set = []
    writer = tf.python_io.TFRecordWriter(set_name)

    image_count = len(images)
    image_counter = 0
    for image in images:
        idx = 0
        #print("progress", test)
        while True:
            if edit_filename(image, idx) not in labels:
                break;
            label = labels[edit_filename(image, idx)]
            if label['color'] == "none":
                idx = idx + 1
                continue
            output_label = format_label(label)
            writer.write(load_image(image, images_path + image, label))
            image_set.append(load_image(image, images_path + image, label))
            idx = idx + 1
        image_counter = image_counter + 1
        if (image_counter % 1000) == 0:
            print("%i images processed, %i percent complete" %(image_counter,
                ceil((image_counter / image_count) * 100)))
    writer.close()



def load_image(image, image_path, label):

    x1 = int(label['x1'])
    y1 = int(label['y1'])
    x2 = int(label['x2'])
    y2 = int(label['y2'])

    output_label = int(format_label(label))
    img = cv2.imread(image_path)
    if img is None:
        return None

    crop_img = img[y1:y2, x1:x2]
    cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    processed_img = cv2.resize(crop_img, (GLOBAL_IMAGE_SIZE, GLOBAL_IMAGE_SIZE))

    # Uncomment below to display resulting image
    # cv2.imshow("light", processed_img)
    # cv2.waitKey(0)

    string_img = _bytes_feature(processed_img.tostring())

    feature = {
        'image_raw': string_img,
        'label': _int64_feature(output_label)
    }

    encoded_data = tf.train.Example(features=tf.train.Features(feature=feature))
    return encoded_data.SerializeToString()

def format_label(label):
    color = label['color']

    conversion_dict = {'red': '0', 'green': '1', 'yellow': '2'}
    return conversion_dict[color]

def get_image_filenames(data_path):
    return listdir(data_path)

def get_labels(labels_path):
    """
    Make a dictionary that holds label data:
        {original-file-name_%idx%.jpg :
                {
                    color:
                    x1:
                    y1:
                    x2:
                    y2:
                    occludeded:
                    truncated:
                }}

    """
    with open(labels_path) as f:
        data = json.load(f)

    # Dataset Summary Stats:
    # Number of traffic lights: 186,117
    # Number of images with multiple traffic lights: 34,660

    # Create a dictionary entry for each traffic light entry
    label_dict = {}
    for picture in data:
        light_idx = 0
        for label in picture['labels']:
            if label['category'] == "traffic light":
                label_dict[edit_filename(picture['name'], light_idx)] = {
                    'color': label['attributes']['trafficLightColor'],
                    'x1': label['box2d']['x1'],
                    'y1': label['box2d']['y1'],
                    'x2': label['box2d']['x2'],
                    'y2': label['box2d']['y2'],
                    'occluded': label['attributes']['occluded'],
                    'truncated': label['attributes']['truncated']
                }
                light_idx = light_idx + 1
    return label_dict

def edit_filename(name, idx):
    no_extension = name[:-4]
    new_filename = no_extension + '_' + str(idx) + '.jpg'
    return new_filename

if __name__ == "__main__":
    extract_data()
