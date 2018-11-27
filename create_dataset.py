import cv2
import csv
from PIL import Image
import tensorflow as tf

GLOBAL_IMAGE_SIZE = 100
IMAGES_TO_PARSE = 5
PREFIX_PATH = "classifier/signDatabasePublicFramesOnly/"

# Used to convert images and annotations to TensorFlow compatible bytestring
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

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
    annotation_tag_string = _bytes_feature(str.encode(annotation_tag))
    feature = {
        'image_raw':string_img,
        'label':annotation_tag_string
        }
    encoded_data = tf.train.Example(features=tf.train.Features(feature=feature))
    return encoded_data.SerializeToString()


if __name__ == "__main__":
    csv_path = "classifier/signDatabasePublicFramesOnly/allAnnotations.csv"
    images = extract_parameters(csv_path)
    for image in images:
        if len(image) == 6: # If in correct format
            # Take the arguments, pass to load_image and print result
            arg0 = image[0]
            arg1 = image[1]
            arg2 = image[2]
            arg3 = image[3]
            arg4 = image[4]
            arg5 = image[5]
            print(load_image(arg0, arg1, arg2, arg3, arg4, arg5))
    # Example of arguments in plain text below:
    #load_image(PREFIX_PATH+"aiua120214-0/frameAnnotations-DataLog02142012_external_camera.avi_annotations/stop_1330545910.avi_image0.png",
    #862, 104, 916, 158, "stop")
