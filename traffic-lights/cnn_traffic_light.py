from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

# Our application logic will be added here


def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  input_layer = tf.reshape(features["x"], [-1, 50, 50, 3])


  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=10,
      kernel_size=[10, 10],
      padding="valid",
      activation=tf.nn.relu)

  # With "valid" padding, this should output a
  # [-1, 40, 40, 10]
  print("conv1")
  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
  # [-1, 20, 20, 10]
  print("pool1")
  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=20,
      kernel_size=[10, 10],
      padding="valid",
      activation=tf.nn.relu)

  print("conv2")
  # With "valid" padding, should output:
  # [-1, 10, 10, 20]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
  print("pool2")
  #[-1, 5, 5, 20]
  # Dense Layer
  pool2_flat = tf.reshape(pool2, [-1, 5 * 5 * 20])
  print("pool2flat")
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  print("dense")
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
  print("dropout")
  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=3)
  print("logits")
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

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def input_fn(filenames):
  print("input_fn")
  dataset = tf.data.TFRecordDataset(filenames=filenames, num_parallel_reads=40)
  # dataset = dataset.apply(
  #     tf.contrib.data.shuffle_and_repeat(1024, 1)
  # )
  dataset = dataset.apply(
      tf.contrib.data.map_and_batch(parser, 32)
  )
  #dataset = dataset.map(parser, num_parallel_calls=12)
  #dataset = dataset.batch(batch_size=1000)
  dataset = dataset.prefetch(buffer_size=2)
  return dataset

def train_input_fn():
  return input_fn(filenames=["train.tfrecords"])

def eval_input_fn():
  return input_fn(filenames=["validation.tfrecords"])

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
  print(image, label)
  return {'x': image}, label


# Load training and eval data
mnist = tf.contrib.learn.datasets.load_dataset("mnist")
train_data = mnist.train.images # Returns np.array

train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
eval_data = mnist.test.images # Returns np.array
eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

# Create the Estimator
mnist_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir="/tmp/whydoihavetokeepchangingthis")

# Set up logging for predictions
tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_iter=50)

# Train the model
# train_input_fn = tf.estimator.inputs.numpy_input_fn(
#   x={"x": train_data},
#   y=train_labels,
#   batch_size=100,
#   num_epochs=None,
#   shuffle=True)
# mnist_classifier.train(
#   input_fn=train_input_fn,
#   steps=1000,
#   hooks=[logging_hook])
#
# # Evaluate the model and print results
# eval_input_fn = tf.estimator.inputs.numpy_input_fn(
#   x={"x": eval_data},
#   y=eval_labels,
#   num_epochs=1,
#   shuffle=False)

mnist_classifier.train(input_fn=train_input_fn, steps=3000)
eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
print(eval_results)
# if __name__ == "__main__":
#   tf.app.run()
