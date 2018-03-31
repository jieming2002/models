#!/usr/bin/env python3
import logging
import os

import cv2
import numpy as np
import tensorflow as tf
from vgg import vgg_16

from object_detection.utils import dataset_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw pet dataset.')
flags.DEFINE_string('output_dir', '', 'Path to directory to output TFRecords.')

FLAGS = flags.FLAGS

classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
           'dog', 'horse', 'motorbike', 'person', 'potted plant',
           'sheep', 'sofa', 'train', 'tv/monitor']
# RGB color for each class
colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
            [128, 0, 128], [0, 128, 128], [
                128, 128, 128], [64, 0, 0], [192, 0, 0],
            [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
            [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
            [0, 192, 0], [128, 192, 0], [0, 64, 128]]


cm2lbl = np.zeros(256**3)
for i, cm in enumerate(colormap):
    cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i


def image2label(im):
    data = im.astype('int32')
    # cv2.imread. default channel layout is BGR
    idx = (data[:, :, 2] * 256 + data[:, :, 1]) * 256 + data[:, :, 0]
    return np.array(cm2lbl[idx])


def dict_to_tf_example(data, label):
    """Convert image to tf.Example proto.

    Args:
        data: the file name of image (*.jpg )
        label: the file name of label (*.png)

    Returns:
        example: The converted tf.Example.
    """
    print('data= ', data)
    # logging.warning('skye here')

    with open(data, 'rb') as inf:
        encoded_data = inf.read()
    img_label = cv2.imread(label)
    img_mask = image2label(img_label)
    encoded_label = img_mask.astype(np.uint8).tobytes()

    height, width = img_label.shape[0], img_label.shape[1]
    if height < vgg_16.default_image_size or width < vgg_16.default_image_size:
        # 
        return None

    # Your code here, fill the dict
    feature_dict = {
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(
          data.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_data),
        'image/label': dataset_util.bytes_feature(
          label.encode('utf8')),
        'image/format': dataset_util.bytes_feature(encoded_label),
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    # print('example= ', example)
    return example


def create_tf_record(output_filename, file_pars):
    # Your code here
    writer = tf.python_io.TFRecordWriter(output_filename)
    for idx, example in enumerate(file_pars):
        # print('example =', example)
        if idx % 100 == 0:
            logging.info('On image %d of %d', idx, len(file_pars))
        
        data_path = example[0]
        label_path = example[1]
        # print('data_path =', data_path)

        if not os.path.exists(data_path):
            logging.warning('Could not find %s, ignoring example.', data_path)
            continue
        
        if not os.path.exists(label_path):
            logging.warning('Could not find %s, ignoring example.', label_path)
            continue
        
        try:
            tf_example = dict_to_tf_example(data_path, label_path)
            writer.write(tf_example.SerializeToString())
        except ValueError:
            logging.warning('Invalid example: %s, ignoring.', data_path)

    writer.close()


def read_images_names(root, train=True):
    txt_fname = os.path.join(root, 'ImageSets/Segmentation/', 'train.txt' if train else 'val.txt')

    with open(txt_fname, 'r') as f:
        images = f.read().split()

    data = []
    label = []
    for fname in images:
        data.append('%s/JPEGImages/%s.jpg' % (root, fname))
        label.append('%s/SegmentationClass/%s.png' % (root, fname))
    return zip(data, label)


def main(_):
    logging.info('Prepare dataset file names')

    train_output_path = os.path.join(FLAGS.output_dir, 'fcn_train.record')
    val_output_path = os.path.join(FLAGS.output_dir, 'fcn_val.record')

    train_files = read_images_names(FLAGS.data_dir, True)
    val_files = read_images_names(FLAGS.data_dir, False)
    create_tf_record(train_output_path, train_files)
    create_tf_record(val_output_path, val_files)


if __name__ == '__main__':
    tf.app.run()
