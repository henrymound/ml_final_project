import cv2
import csv
from PIL import Image
import tensorflow as tf
from random import shuffle
import sys
import numpy as np
import glob

GLOBAL_IMAGE_SIZE = 224
IMAGES_TO_PARSE = 10000000
PREFIX_PATH = "classifier/signDatabasePublicFramesOnly/"

# Used to convert images and annotations to TensorFlow compatible bytestring
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# Takes in a line from the csv file and parses the included values.
#   Returns a list. Each element in the list corresponds to an image from the csv.
#   Each element in the list is an array of paramers in the correct order to
#   be processed by load_image
def extract_parameters(csv_path):
    list_to_return = list()
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count != 0: # Don't process first line
                # Split row into an array of strings based on semicolon placement
                #   (as outlined in the dataset)
                first_parameter_split = row[0].split(";")
                filename = first_parameter_split[0]
                annotation_tag = first_parameter_split[1]
                upper_left_x = first_parameter_split[2]
                upper_left_y = first_parameter_split[3]
                lower_left_x = first_parameter_split[4]
                lower_left_y = first_parameter_split[5]
                occluded = first_parameter_split[5]
                array_of_parameters = [
                    PREFIX_PATH+filename,
                    int(upper_left_x),
                    int(upper_left_y),
                    int(lower_left_x),
                    int(lower_left_y),
                    annotation_tag]
                list_to_return.append(array_of_parameters)
            line_count+=1
            if line_count >= IMAGES_TO_PARSE: # Only do the first 25 lines, for debugging
                break;
        return list_to_return
        #print(f'Processed {line_count} lines.')


def load_image(addr, x1, y1, x2, y2, annotation_tag):
    # Read an image, crop it to bounds as defined by parameters
    #   and resize the resulting image (200, 200).
    #   Return a serialized feature including the image in string form
    #   as well as the image's annotation

    label = annotation_to_label(annotation_tag)

    img = cv2.imread(addr)
    if img is None: # If the path cant be found, return None
        return None
    crop_img = img[y1:y2, x1:x2] # crop image
    cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB) # convert to color
    # resize image to global
    processed_img = cv2.resize(crop_img, (GLOBAL_IMAGE_SIZE, GLOBAL_IMAGE_SIZE))

    # Uncomment below to display resulting image
    #v2.imshow("sign", processed_img)
    #cv2.waitKey(0)

    # Convert image to string
    string_img = _bytes_feature(processed_img.tostring())

    # Convert tag to bytes and then convert it into a feature
    feature = {
        'image_raw':string_img,
        'label': _int64_feature(label)
        }
    encoded_data = tf.train.Example(features=tf.train.Features(feature=feature))
    return encoded_data.SerializeToString()


def output_data_record(csv_path, filename, imageset):
    writer = tf.python_io.TFRecordWriter(filename)

    for image in imageset:
        img = load_image(image[0], image[1], image[2], image[3], image[4], image[5])
        writer.write(img)

    writer.close()

def annotation_to_label(annotation):
    conversions = {
        'addedLane': 0,
        'curveLeft': 1,
        'curveRight': 2,
        'dip': 3,
        'doNotEnter': 4,
        'doNotPass': 5,
        'intersection': 6,
        'keepRight': 7,
        'laneEnds': 8,
        'merge': 9,
        'noLeftTurn': 10,
        'noRightTurn': 11,
        'pedestrianCrossing': 12,
        'rampSpeedAdvisory20': 13,
        'rampSpeedAdvisory35': 14,
        'rampSpeedAdvisory40': 15,
        'rampSpeedAdvisory45': 16,
        'rampSpeedAdvisory50': 17,
        'rampSpeedAdvisoryUrdbl': 18,
        'rightLaneMustTurn': 19,
        'roundabout': 20,
        'school': 21,
        'schoolSpeedLimit25': 22,
        'signalAhead': 23,
        'slow': 24,
        'speedLimit15': 25,
        'speedLimit25': 26,
        'speedLimit30': 27,
        'speedLimit35': 28,
        'speedLimit40': 29,
        'speedLimit45': 30,
        'speedLimit50': 31,
        'speedLimit55': 32,
        'speedLimit65': 33,
        'speedLimitUrdbl': 34,
        'stop': 35,
        'stopAhead': 36,
        'thruMergeLeft': 37,
        'thruMergeRight': 38,
        'thruTrafficMergeLeft': 39,
        'truckSpeedLimit55': 40,
        'turnLeft': 41,
        'turnRight': 42,
        'yield': 43,
        'yieldAhead': 44,
        'zoneAhead25': 45,
        'zoneAhead45': 46
    }
    return conversions[annotation]


if __name__ == "__main__":
    csv_path = "classifier/signDatabasePublicFramesOnly/allAnnotations.csv"
    images = extract_parameters(csv_path) # 7855 images
    images_shuffled = shuffle(images)

    # Divide the data into 60% train, 20% validation, 20% test
    output_data_record(csv_path, 'train.tfrecords', images[0:4713])
    output_data_record(csv_path, 'val.tfrecords', images[4713:6284])
    output_data_record(csv_path, 'test.tfrecords', images[6284:7855])



    # for image in images:
    #     if len(image) == 6: # If in correct format
    #         # Take the arguments, pass to load_image and print result
    #         arg0 = image[0]
    #         arg1 = image[1]
    #         arg2 = image[2]
    #         arg3 = image[3]
    #         arg4 = image[4]
    #         arg5 = image[5]
    #         print(load_image(arg0, arg1, arg2, arg3, arg4, arg5))
    # Example of arguments in plain text below:
    #load_image(PREFIX_PATH+"aiua120214-0/frameAnnotations-DataLog02142012_external_camera.avi_annotations/stop_1330545910.avi_image0.png",
    #862, 104, 916, 158, "stop")
