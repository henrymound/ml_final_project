import cnn_traffic_light as cnn
import sys

import numpy as np
import tensorflow as tf
from cnn_traffic_light import input_fn
from tensorflow.contrib import predictor
import cv2
import argparse
import numpy as np
import tensorflow as tf

global COLORS

GLOBAL_IMAGE_SIZE = 50

# Used to convert images and annotations to TensorFlow compatible bytestring
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h, classes):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def segment_image(image_path):
    arg_config = "yolov3.cfg"
    arg_weights = "yolov3.weights"
    arg_classes = "yolov3.txt"

    image = cv2.imread(image_path)
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392
    classes = None

    with open(arg_classes, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    global COLORS
    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
    net = cv2.dnn.readNet(arg_weights, arg_config)
    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(get_output_layers(net))
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])


    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    traffic_light_images = list()

    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        if class_ids[i] == 9: # If one of the detected objects is a traffic light
            y2 = int(y + h)
            x2 = int(x + w)
            crop_img = image[int(y):y2, int(x):x2]
            traffic_light_images.append(crop_img)
            # Uncomment below to display the cropped images of traffic lights
            # cv2.imshow("cropped", crop_img)
            # cv2.waitKey()

    print(str(len(traffic_light_images)) + " traffic lights found!")
    encoded_traffic_light_images = list()
    string_images = []
    # Loop through the found traffic lights and process them
    for traffic_light_image in traffic_light_images:

        cv2.cvtColor(traffic_light_image, cv2.COLOR_BGR2RGB)
        processed_img = cv2.resize(traffic_light_image, (GLOBAL_IMAGE_SIZE, GLOBAL_IMAGE_SIZE))
        string_img = _bytes_feature(processed_img.tostring())
        string_images.append(string_img)
        feature = {
            'image_raw': string_img
        }
        encoded_data = tf.train.Example(features=tf.train.Features(feature=feature))
        encoded_traffic_light_images.append(encoded_data)
    return encoded_traffic_light_images
    #print(encoded_traffic_light_images)
    #draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h), classes)

    # cv2.imshow("object detection", image)
    # cv2.waitKey()
    #
    # cv2.imwrite("object-detection.jpg", image)
    # cv2.destroyAllWindows()

def make_color_prediction(encoded_image_list):
    full_model_dir = 'model2/test/'
    with tf.Session() as sess:
        predict_fn = predictor.from_saved_model('output_model/1544369743')
        #model_input = tf.train.Example(features=tf.train.Features( feature={"words": tf.train.Feature(int64_list=tf.train.Int64List(value=features_test_set)) }))
        #model_input = model_input.SerializeToString()
        predictions = predict_fn({"x": encoded_image_list[0]}.SerializeToString())

if __name__=="__main__":
    traffic_lights=segment_image(sys.argv[1])
    new_traffic_lights = []
    for idx, img in enumerate(traffic_lights):
        new_traffic_lights.append(cnn.prod_parse(img))
    make_color_prediction(traffic_lights)
