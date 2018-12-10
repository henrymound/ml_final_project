from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf


def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""


  # Input Layer - (50x50 pixels and 3 color channels)
  input_layer = tf.reshape(features["x"], [-1, 50, 50, 3])

  # Convolutional Layer #1 (20 channels, 10x10 filter, same padding, stride=1)
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=20,
      kernel_size=[10, 10],
      padding="same",
      activation=tf.nn.relu)

  # Pool layer 1: 2x2 pool size, strides=5
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=5)


  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=40,
      kernel_size=[2, 2],
      padding="same",
      activation=tf.nn.relu)

  # With "valid" padding, should output:
  # [-1, 10, 10, 20]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Unroll the layer into a single column
  pool2_flat = tf.reshape(pool2, [-1, 5 * 5 * 40])

  # Fully connected layer (1024 units)
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

  # Implement dropout at rate of 0.3 (30% nodes dropped each time)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.3, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits/Output layer
  logits = tf.layers.dense(inputs=dropout, units=4)
  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Training configuration (minimize via gradient descent with alpha = .001)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Evaluation metrics
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def input_fn(filenames):
  """Configure the dataset for use within the model."""
  dataset = tf.data.TFRecordDataset(filenames=filenames, num_parallel_reads=40)
  # dataset = dataset.apply(
  #     tf.contrib.data.shuffle_and_repeat(1024, 1)
  # )
  dataset = dataset.apply(
      tf.contrib.data.map_and_batch(parser, 32)
  )

  dataset = dataset.prefetch(buffer_size=2)
  return dataset

def train_input_fn():
  """Return the input_fn function with the training output file name as its argument"""
  return input_fn(filenames=["train_none.tfrecords"])

def eval_input_fn():
  """Return the input_fn function with the validation output file name as its argument"""
  return input_fn(filenames=["validation_none.tfrecords"])

def parser(record):
  """Called by input_fn to format examples properly"""
  keys_to_features = {
      "image_raw": tf.FixedLenFeature([], tf.string),
      "label":     tf.FixedLenFeature([], tf.int64)
  }
  parsed = tf.parse_single_example(record, keys_to_features)
  image = tf.decode_raw(parsed["image_raw"], tf.uint8)
  image = tf.cast(image, tf.float32)
  label = tf.cast(parsed["label"], tf.int32)
  print({'x': image})
  return {'x': image}, label

def prod_parse(record):
  """Helper function to be called by a script passing input to this model for
     predictions"""
  keys_to_features = {
      "image_raw": tf.FixedLenFeature([], tf.string),
  }
  parsed = tf.parse_single_example(record, keys_to_features)
  image = tf.decode_raw(parsed["image_raw"], tf.uint8)
  image = tf.cast(image, tf.float32)
  return image


def serving_input_receiver_fn():
  """Build the serving inputs for predictions (not training or validation)"""
  inputs = {"x": tf.placeholder(dtype=tf.float32)}
  return tf.estimator.export.ServingInputReceiver(inputs, inputs)


def main():
    """Create, train, and save the network"""

    tf.logging.set_verbosity(tf.logging.INFO)
    
    # Create the Estimator
    classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="model2/test")

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    # Training
    i = 0
    while i < 1:
        classifier.train(input_fn=train_input_fn, steps=1)
        eval_results = classifier.evaluate(input_fn=eval_input_fn)
        print(eval_results)
        i = i + 1

    # Export the network
    export_dir = classifier.export_savedmodel(
        export_dir_base="output_model",
        serving_input_receiver_fn=serving_input_receiver_fn)


if __name__ == "__main__":
  main()
