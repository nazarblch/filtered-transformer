import torch
import tensorflow as tf

tfrecord_path = "/home/nazar/valid-0-0.tfr"

# Create a description of the features.
feature_description = {
    'features': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'labels': tf.io.FixedLenFeature([], tf.int64, default_value=0)
}

def _parse_function(example_proto):
  # Parse the input `tf.train.Example` proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, feature_description)


raw_dataset = tf.data.TFRecordDataset([tfrecord_path])
parsed_dataset = raw_dataset.map(_parse_function)

for parsed_record in parsed_dataset.take(10):
  print(repr(parsed_record))
