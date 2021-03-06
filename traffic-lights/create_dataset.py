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


def extract_data():
    """ Extra Berkeley Deep Drive training/validation examples and export them
        as TensorFlow-compatible examples in .tfrecords files"""

    print("\n****Berkeley Deep Drive Data Extraction****\n")
    print("Using global image size of %i x %i pixels" %(GLOBAL_IMAGE_SIZE, GLOBAL_IMAGE_SIZE) )

    print("**Gathering labels for training set:**")
    labels_dict = get_labels(LABELS_TRAIN_PATH)
    print("%i labels found" %len(labels_dict))
    print("*Collecting image filenames*")
    train_images = get_image_filenames(IMAGES_TRAIN_PATH)
    print("*Formatting and encoding images and labels*")
    prep_image_set('train_none.tfrecords', train_images, IMAGES_TRAIN_PATH, labels_dict)
    print("*Training data extracted*\n")

    print("**Gathering labels for validation set:**")
    validation_labels_dict = get_labels(LABELS_VALIDATION_PATH)
    print("%i labels found" %len(validation_labels_dict))
    print("*Collecting image filenames*")
    validation_images = get_image_filenames(IMAGES_VALIDATION_PATH)
    print("*Formatting and encoding images and labels*")
    prep_image_set('validation_none.tfrecords', validation_images,
        IMAGES_VALIDATION_PATH, validation_labels_dict)
    print("*Validation data extracted*")

    print("\n*** Data Extraction Complete ***")
    print("Use train.tfrecords and validation.tfrecords")

"""Used to convert images and annotations to TensorFlow compatible bytestrings"""
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def prep_image_set(set_name, images, images_path, labels):
    """Iterate through a set of images and write an output file with examples
    and labels for each instance of traffic signs in TensorFlow-compative format
    in a .tfrecords file named set_name"""
    image_set = []
    writer = tf.python_io.TFRecordWriter(set_name)

    image_counter = 0 # Number of images processed
    for image in images:
        idx = 0
        while True:
            if edit_filename(image, idx) not in labels:
                break;
            label = labels[edit_filename(image, idx)]
            output_label = format_label(label)
            try:
                output = load_image(image, images_path + image, label)
                writer.write(output)
            except:
                break
            image_set.append(load_image(image, images_path + image, label))
            idx = idx + 1
        image_counter = image_counter + 1
        if (image_counter % 1000) == 0:
            print("%i images processed, %i percent complete" %(image_counter,
                ceil((image_counter / image_count) * 100)))
    writer.close()



def load_image(image, image_path, label):
    """Crop intput image according to label and global image size. Then encode
    it in proper format. Called by prep_image_set()"""
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
    """Return numerical label for light color given annotation as input"""
    color = label['color']

    conversion_dict = {'red': '0', 'green': '1', 'yellow': '2', 'none': '3'}
    return conversion_dict[color]

def get_image_filenames(data_path):
    """Get list of all image names in file (training/validation sets are in
    different files)"""
    return listdir(data_path)

def get_labels(labels_path):
    """Load image label file and create a dictionary of labels for each image
    name as key."""
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
    """Change file extension to account for multiple traffic light examples per
        file"""
    no_extension = name[:-4]
    new_filename = no_extension + '_' + str(idx) + '.jpg'
    return new_filename

if __name__ == "__main__":
    extract_data()
