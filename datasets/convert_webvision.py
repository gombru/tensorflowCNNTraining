# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Downloads and converts Flowers data to TFRecords of TF-Example protos.

This module downloads the Flowers data, uncompresses it, reads the files
that make up the Flowers data and creates two TFRecord datasets: one for train
and one for test. Each TFRecord dataset is comprised of a set of TF-Example
protocol buffers, each of which contain a single image and label.

The script should take about a minute to run.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys

from datasets import dataset_utils

import tensorflow as tf

# Seed for repeatability.
_RANDOM_SEED = 0

# The number of shards per dataset split.
_NUM_SHARDS = 20


class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


def _get_filenames_and_classes(dataset_dir, split):

  photo_filenames = []
  photo_labels = []

  prefix = ''
  if split.__contains__('val'):
      prefix = 'val_images_256/'

  with open(dataset_dir + split, 'r') as annsfile:
      for l in annsfile:
          data = l.split(' ')
          photo_filenames.append(dataset_dir + prefix + data[0])
          photo_labels.append(int(data[1]))

  return photo_filenames, photo_labels


def _get_dataset_filename(dataset_dir, split_name, shard_id):
  output_filename = 'webvision_%s_%05d-of-%05d.tfrecord' % (
      split_name, shard_id, _NUM_SHARDS)
  return os.path.join('/home/Imatge/ssd2/WebVision/TF-dataset/', output_filename)


def _convert_dataset(split_name, filenames, labels, dataset_dir):

  num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))

  with tf.Graph().as_default():
    image_reader = ImageReader()

    with tf.Session('') as sess:

      for shard_id in range(_NUM_SHARDS):
        output_filename = _get_dataset_filename(dataset_dir, split_name, shard_id)

        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
          start_ndx = shard_id * num_per_shard
          end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
          for i in range(start_ndx, end_ndx):
            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (i+1, len(filenames), shard_id))
            sys.stdout.flush()

            # Read the filename:
            image_data = tf.gfile.FastGFile(filenames[i], 'rb').read()
            height, width = image_reader.read_image_dims(sess, image_data)
            #class_name = os.path.basename(os.path.dirname(filenames[i]))
            class_id = int(labels[i])

            example = dataset_utils.image_to_tfexample(
                image_data, b'jpg', height, width, class_id)
            tfrecord_writer.write(example.SerializeToString())

  sys.stdout.write('\n')
  sys.stdout.flush()





def _dataset_exists(dataset_dir):
  for split_name in ['train', 'validation']:
    for shard_id in range(_NUM_SHARDS):
      output_filename = _get_dataset_filename(
          dataset_dir, split_name, shard_id)
      if not tf.gfile.Exists(output_filename):
        return False
  return True


def run(dataset_dir, train, val):
  """Runs the  conversion operation.

  Args:
    dataset_dir: The dataset directory where the dataset is stored.
  """

  if _dataset_exists(dataset_dir):
    print('Dataset files already exist. Exiting without re-creating them.')
    return

  # Validation
  photo_filenames, photo_labels = _get_filenames_and_classes(dataset_dir, val)
  print('Validation images: ' + str(len(photo_filenames)))
  random.seed(_RANDOM_SEED)
  c = list(zip(photo_filenames, photo_labels))
  random.shuffle(c)
  photo_filenames, photo_labels = zip(*c)

  _convert_dataset('validation', photo_filenames, photo_labels, dataset_dir)
  print('Validation set conversion done')

  # Train
  photo_filenames, photo_labels = _get_filenames_and_classes(dataset_dir, train)
  print('Training images: ' + str(len(photo_filenames)))
  random.seed(_RANDOM_SEED)
  c = list(zip(photo_filenames, photo_labels))
  random.shuffle(c)
  photo_filenames, photo_labels = zip(*c)
  _convert_dataset('train', photo_filenames, photo_labels, dataset_dir)
  print('Training set conversion done')


  # Finally, write the labels file:
  # labels_to_class_names = dict(zip(range(len(class_names)), class_names))
  # dataset_utils.write_label_file(labels_to_class_names, dataset_dir)

  print('\nFinished converting the WebVision dataset!')


run('../../datasets/WebVision/', 'info/train_filtered_balanced.txt','info/val_filelist.txt')