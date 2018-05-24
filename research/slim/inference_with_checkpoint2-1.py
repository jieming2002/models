"""Generic evaluation script that evaluates a model using a given image.
and copy the image to the label dir. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory
import numpy as np
import os
import shutil


tf.app.flags.DEFINE_string(
    'dataset_name', 'cifar10', 'The name of the dataset to load.')
tf.app.flags.DEFINE_string(
    'dataset_dir', '/home/skye/tmp/cifar10', 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_string(
    'model_name', 'cifarnet', 'The name of the architecture to evaluate.')

# herw we cannot use ~/tmp, must use full directory as /home/skye/tmp
tf.app.flags.DEFINE_string(
    'checkpoint_path', '/home/skye/tmp/cifarnet-model', 'The directory of checkpoint.')

tf.app.flags.DEFINE_string(
    'pic_path', './test.jpeg', 'The name of the image.')

tf.app.flags.DEFINE_string(
    'label_path', '', 'The path of the label file.')

tf.app.flags.DEFINE_integer(
    'default_image_size', 300, 'The size of the image.')

FLAGS = tf.app.flags.FLAGS

is_training = False
preprocessing_name = FLAGS.model_name


# get label of model predict test image top_k predict
def get_label_predict_top_k(logits, labels, top_k):
    """
    logits : an array
    labels : a dict of index to class name
    return top-5 of label name
    """
    # array 2 list
    predict_list = list(logits[0][0])
    # print('len(predict_list) =', len(predict_list))
    # print('predict_list =', predict_list)
    
    min_label = min(predict_list)
    label_k = ''
    for i in range(top_k):
        label = np.argmax(predict_list)
        predict_list.remove(predict_list[label])
        predict_list.insert(label, min_label)
        label_name = labels[label]
        label_k += label_name
    return label_k


graph = tf.Graph().as_default()

# just use dataset to give the num_classes
dataset = dataset_factory.get_dataset(FLAGS.dataset_name, 'train', FLAGS.dataset_dir)

image_preprocessing_fn = preprocessing_factory.get_preprocessing(
    preprocessing_name,
    is_training=is_training
)

network_fn = nets_factory.get_network_fn(
    FLAGS.model_name,
    num_classes=(dataset.num_classes),
    is_training=is_training
)

if hasattr(network_fn, 'default_image_size'):
    image_size = network_fn.default_image_size
else:
    image_size = FLAGS.default_image_size

with open(FLAGS.label_path, encoding='utf-8') as inf:
    labels_to_class_names = {}
    for line in inf:
        cols = line.strip().split(":")
        labels_to_class_names[int(cols[0])] = cols[1]
    # print('labels_to_class_names =', labels_to_class_names)

# begin setting graph
placeholder = tf.placeholder(name='input', dtype=tf.string)
# if the image is png, then the channel will be 4, so we must set channels=3
image = tf.image.decode_jpeg(placeholder, channels=3)
image = image_preprocessing_fn(image, image_size, image_size)
# to match the dimensions of cifarnet, we must expand the 0 dimension as the num of input
image = tf.expand_dims(image, 0)
logit, end_points = network_fn(image)

saver = tf.train.Saver()
with tf.Session() as sess:
    # print('FLAGS.checkpoint_path = ', FLAGS.checkpoint_path)
    checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    # print('checkpoint_path = ', checkpoint_path)
    # this method is more slow then output method
    saver.restore(sess, checkpoint_path)
    
    image_name_list = []
    predict_labels = []
    num_images = 0

    for filename in os.listdir(FLAGS.pic_path):
        # load the image
        path = os.path.join(FLAGS.pic_path, filename)
        if not os.path.isdir(path):
            with open(path, mode='rb') as f_img:
                image_value = f_img.read()

                logit_value = sess.run([logit], feed_dict={placeholder:image_value})
                # print('logit_value = ', logit_value)
                top_5 = get_label_predict_top_k(logit_value, labels_to_class_names, 5)
                
                # print('top_5 =', top_5)
                path_label = os.path.join(FLAGS.pic_path, top_5[0])
                # print('path_label = ', path_label)
                if not tf.gfile.Exists(path_label):
                    tf.gfile.MakeDirs(path_label)
                
                target_path = os.path.join(path_label, filename)
                # print('target_path = ', target_path)
                shutil.copyfile(path, target_path)

                # image_name_list.append(filename)
                # predict_labels.append(top_5)
                # print(filename, top_5)
                num_images += 1
                print('num_images =', num_images)
            # break
    
    # np.save('./tmp/image_name_list', image_name_list)
    # np.save('./tmp/predict_labels', predict_labels)
    # print('image_name_list =', image_name_list)
    # print('predict_labels =', predict_labels)
