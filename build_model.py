import tensorflow as tf
import cv2
import sys
import numpy as np

## Adapted from https://github.com/kalaspuffar/tensorflow-data


# Load training and eval data
sess = tf.Session()
sess.run(tf.global_variables_initializer())


def train_input_fn():
    return input_fn(filenames=["train.tfrecords", "test.tfrecores"])

def val_input_fn():
    return input_fn(filenames=["val.tfrecords"])

def parser(record):
    keys_to_features = {
        "image_raw": tf.FixedLenFeature([], tf.string),
        "label":     tf.FixedLenFeature([], tf.int64)
    }
    parsed = tf.parse_single_example(record, keys_to_features)
    image = tf.decode_raw(parsed["image_raw"], tf.uint8)
    image = tf.cast(image, tf.float32)
    #image = tf.reshape(image, shape=[224, 224, 3])
    label = tf.cast(parsed["label"], tf.int32)

    return {'image': image}, label

def input_fn(filenames):
  dataset = tf.data.TFRecordDataset(filenames=filenames, num_parallel_reads=40)
  dataset = dataset.apply(
      tf.contrib.data.shuffle_and_repeat(1024, 1)
  )
  dataset = dataset.apply(
      tf.contrib.data.map_and_batch(parser, 32)
  )
  #dataset = dataset.map(parser, num_parallel_calls=12)
  #dataset = dataset.batch(batch_size=1000)
  dataset = dataset.prefetch(buffer_size=2)
  return dataset

def model_fn(features, labels, mode, params):
    """Model function for CNN"""

    num_classes = 47

    net = features["image"]

    net = tf.identity(net, name="input_tensor")

    net = tf.reshape(net, [-1, 100, 100, 3])

    net = tf.identity(net, name="input_tensor_after")

    net = tf.layers.conv2d(inputs=net, name='layer_conv1',
                           filters=32, kernel_size=3,
                           padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

    net = tf.layers.conv2d(inputs=net, name='layer_conv2',
                           filters=64, kernel_size=3,
                           padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

    net = tf.layers.conv2d(inputs=net, name='layer_conv3',
                           filters=64, kernel_size=3,
                           padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

    net = tf.contrib.layers.flatten(net)

    net = tf.layers.dense(inputs=net, name='layer_fc1',
                        units=128, activation=tf.nn.relu)

    net = tf.layers.dropout(net, rate=0.5, noise_shape=None,
                        seed=None, training=(mode == tf.estimator.ModeKeys.TRAIN))

    net = tf.layers.dense(inputs=net, name='layer_fc_2',
                        units=num_classes)

    logits = net
    y_pred = tf.nn.softmax(logits=logits)

    y_pred = tf.identity(y_pred, name="output_pred")

    y_pred_cls = tf.argmax(y_pred, axis=1)

    y_pred_cls = tf.identity(y_pred_cls, name="output_cls")


    if mode == tf.estimator.ModeKeys.PREDICT:
        spec = tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=y_pred_cls)
    else:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                                       logits=logits)
        loss = tf.reduce_mean(cross_entropy)

        optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"])
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())
        metrics = {
            "accuracy": tf.metrics.accuracy(labels, y_pred_cls)
        }

        spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=metrics)

    return spec


model = tf.estimator.Estimator(
   model_fn=model_fn, params={"learning_rate": 1e-4}, model_dir="./model5/"
 )
count = 0
while (count < 100000):
    model.train(input_fn=train_input_fn, steps=100)
    result = model.evaluate(input_fn=val_input_fn)
    print(result)
    print("Classification accuracy: {0:.2%}".format(result["accuracy"]))
    sys.stdout.flush()
    count = count + 1



    # Input layer
    # input_layer = tf.reshape(features["image"], [-1, 112, 112, 3])
    #
    # # Convolutional Layer #1
    # conv1 = tf.layers.conv2d(
    #     inputs=input_layer,
    #     filters=32,
    #     kernel_size=[4,4],
    #     padding="same",
    #     activation=tf.nn.relu)
    #
    # pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[4, 4], strides=2)
    #
    # # Convolutional Layer #2 and Pooling Layer #2
    # conv2 = tf.layers.conv2d(
    #     inputs=pool1,
    #     filters=64,
    #     kernel_size=[4, 4],
    #     padding="same",
    #     activation=tf.nn.relu)
    # pool2 = tf.layers.max_pooling2d(inputs=conv2,pool_size=[5, 5],strides=2)
    #
    # #Dense Layer
    # pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    # dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    # dropout = tf.layers.dropout(
    #   inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    #
    # # Logits Layer
    # logits = tf.layers.dense(inputs=dropout, units=47)

    # predictions = {
    #     # Generate predictions (for PREDICT and EVAL mode)
    #     "classes": tf.argmax(input=logits, axis=1),
    #     # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
    #     # `logging_hook`.
    #     "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    #     }
    # if mode == tf.estimator.ModeKeys.PREDICT:
    #     return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    #
    # # Calculate Loss (for both TRAIN and EVAL modes)
    # loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    #
    # # Configure the Training Op (for TRAIN mode)
    # if mode == tf.estimator.ModeKeys.TRAIN:
    #   optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    #   train_op = optimizer.minimize(
    #     loss=loss,
    #     global_step=tf.train.get_global_step())
    #   return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    #
    # # Add evaluation metrics (for EVAL mode)
    # eval_metric_ops = {
    #   "accuracy": tf.metrics.accuracy(
    #     labels=labels, predictions=predictions["classes"])}
    # return tf.estimator.EstimatorSpec(
    #   mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
