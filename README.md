# Traffic Sign Segmentation and Classification
## ML Final Project CS451
### Ben Vandenbosch and Henry Mound

In order for this code to run, the user must have Pillow, TensorFlow, and OpenCV2 to run.
create_dataset.py reads from the Berkeley DeepDrive dataset which can be downloaded [here](https://deepdrive.berkeley.edu/#main-menu).
In order for the code to run, the bdd100k_images and bdd100k_images folders should be in the traffic-lights
directory.

To run an image through yolo_opencv.py, use the following format:
```
python yolo_opencv.py --image IMAGE.jpg --config yolov3.cfg --weights yolov3.weights --classes yolov3.txt
```

Credit to:
  - [Arun Ponnusamy](https://github.com/arunponnusamy/object-detection-opencv)
  - [Daniel Kalaspuffar](https://github.com/kalaspuffar/tensorflow-data)
