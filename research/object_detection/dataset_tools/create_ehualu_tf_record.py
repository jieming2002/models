
r"""Convert the ehualu dataset to TFRecord for object_detection.

Example usage:
    python object_detection/dataset_tools/create_pet_tf_record.py \
        --data_dir=/home/user/pet \
        --output_dir=/home/user/pet/output
"""

import hashlib
import io
import logging
import os
import sys
import re
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import PIL.Image
import tensorflow as tf

from object_detection.utils import dataset_util
# from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw dataset.')
flags.DEFINE_string('group', 'train_b', 'group of raw dataset.')
flags.DEFINE_string('output_dir', '', 'Path to directory to output TFRecords.')
flags.DEFINE_string('label_map_path', 'data/pet_label_map.pbtxt',
                    'Path to label map proto')
FLAGS = flags.FLAGS


def dict_to_tf_example(data,
                       label_map_dict,
                       image_subdirectory):
  """Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: a dataframe record holding a single image name and annotations
    label_map_dict: A map from string label names to integers ids.
    image_subdirectory: String specifying subdirectory within the
      Pascal dataset directory holding the actual image data.
    ignore_difficult_instances: Whether to skip difficult instances in the
      dataset  (default: False).

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  filename, annotations = data
  img_path = os.path.join(image_subdirectory, filename)
  # print('img_path=', img_path)
  with tf.gfile.GFile(img_path, 'rb') as fid:
    encoded_jpg = fid.read()
  
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')
  key = hashlib.sha256(encoded_jpg).hexdigest()
  width, height = image.size

  xmins = []
  ymins = []
  xmaxs = []
  ymaxs = []
  classes = []
  classes_text = []
  truncated = []
  poses = []
  difficult_obj = []

  annotation_list = annotations.split(';')
  for obj in annotation_list:
    difficult_obj.append(0)
    # x, y, width, height
    bbox = obj.split('_')
    bbox = [int(float(bbox[0])), int(float(bbox[1])), int(float(bbox[2])), int(float(bbox[3]))]
    # print('bbox =', bbox)
    
    bbox[2] += bbox[0]
    bbox[3] += bbox[1]
    # xmin, ymin, xmax, ymax
    xmin, ymin, xmax, ymax = bbox

    xmins.append(xmin / width)
    ymins.append(ymin / height)
    xmaxs.append(xmax / width)
    ymaxs.append(ymax / height)

    classes_text.append('vv'.encode('utf8'))
    classes.append(0)
    truncated.append(0)
    poses.append('pose'.encode('utf8'))
    
  feature_dict = {
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
      'image/object/truncated': dataset_util.int64_list_feature(truncated),
      'image/object/view': dataset_util.bytes_list_feature(poses),
  }

  # print('feature_dict =', feature_dict)
  example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
  return example


def create_tf_record(output_filename,
                     label_map_dict,
                     image_dir,
                     examples):
  """Creates a TFRecord file from examples.

  Args:
    output_filename: Path to where output file is saved.
    label_map_dict: The label map dictionary.
    image_dir: Directory where image files are stored.
    examples: Examples to parse and save to tf record.
  """
  writer = tf.python_io.TFRecordWriter(output_filename)
  total = examples.shape[0]
  for i in range(total):
    info = examples.iloc[i:i+1]
    # print('info =', info)
    name = info.name.values[0]
    # print('%s = %s' % (i, name))
    coordinate = info.coordinate.values[0]
    # print('coordinate =', coordinate)
    sys.stdout.write('\r>> On image %s/%s %s' % (i, total, name))
    sys.stdout.flush()
    # skip the rows contain no target  
    if len(coordinate) < 1:  
      print('%s==%s no coordinate' % (i,name))
      continue
    try:
      tf_example = dict_to_tf_example(
          [name, coordinate],
          label_map_dict,
          image_dir)
      writer.write(tf_example.SerializeToString())
    except ValueError:
      logging.warning('Invalid example: %s, ignoring.', name)
    # break
  writer.close()
  sys.stdout.write('\n')
  sys.stdout.flush()


# TODO: Add test for pet/PASCAL main files.
def main(_):
  data_dir = FLAGS.data_dir
  # label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
  label_map_dict = {}
  logging.info('Reading from dataset.')
  image_dir = os.path.join(data_dir, FLAGS.group)
  annotations_path = os.path.join(data_dir, '%s.csv' % FLAGS.group)
  annotations_info = pd.read_csv(annotations_path)
  train_examples, val_examples = train_test_split(annotations_info, test_size=0.1, random_state=42)
  # print('num_examples=%s  num_train=%s' % (train_examples.shape, val_examples.shape))
  logging.info('%s training and %s validation examples.', train_examples.shape, val_examples.shape)

  train_output_path = os.path.join(FLAGS.output_dir, '%s_train.record'%FLAGS.group)
  val_output_path = os.path.join(FLAGS.output_dir, '%s_val.record'%FLAGS.group)
  
  create_tf_record(
      train_output_path,
      label_map_dict,
      image_dir,
      train_examples)
  create_tf_record(
      val_output_path,
      label_map_dict,
      image_dir,
      val_examples)


if __name__ == '__main__':
  tf.app.run()
