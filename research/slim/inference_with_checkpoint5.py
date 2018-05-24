""" loop the dir, 
inference and check low/wrong result 
and copy them to a single dir 
then delete the wrong image """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory
import numpy as np
import pandas as pd
import scipy.misc
import matplotlib.pyplot as plt
import shutil
import os


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
    predict_k = []
    for i in range(top_k):
        label = np.argmax(predict_list)
        predict_k.append(predict_list[label])
        predict_list.remove(predict_list[label])
        predict_list.insert(label, min_label)
        label_name = labels[label]
        label_k += label_name
    return label_k, predict_k[0]


def check_result(fpath, path_wrong, path_low, filename, y_label, top_5, top1_p):
    ''' check the result of inference 
    if the result is wrong, then copy the image to a directory.
    else if the probability is < 9.5 , then copy the image to a directory.
    '''
    # reimg = tf.image.decode_jpeg(image_data, channels=1) 
    if(top_5[0] != y_label):
        to_path = os.path.join(path_wrong, top_5+filename)
        if not tf.gfile.Exists(to_path):
            shutil.copyfile(fpath, to_path)
        os.remove(fpath) #delete the wrong image
    elif (top1_p < 9.5):
        # shutil.copyfile(fpath, os.path.join(path_low, filename))
        pass


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
    
    # image_name_list = []
    # predict_labels = []
    # predict_prob = []
    num_images = 0

    for filename in os.listdir(FLAGS.pic_path):
        # load the image
        path = os.path.join(FLAGS.pic_path, filename)
        y_label = filename

        if os.path.isdir(path):
            path_wrong = os.path.join(path, filename+'_wrong')
            path_low = os.path.join(path, filename+'_low')
            # print('path_wrong = ', path_wrong)
            # break

            if not tf.gfile.Exists(path_wrong):
                tf.gfile.MakeDirs(path_wrong)
            
            if not tf.gfile.Exists(path_low):
                tf.gfile.MakeDirs(path_low)
            
            for filename in os.listdir(path):
                fpath = os.path.join(path, filename)
                if not os.path.isdir(fpath):
                    with open(fpath, mode='rb') as f_image:
                        image_value = f_image.read()
                        # image_value = tf.gfile.FastGFile(fpath, 'rb').read()
                        logit_value = sess.run([logit], feed_dict={placeholder:image_value})
                        # print('logit_value = ', logit_value)
                        top_5, top1_p = get_label_predict_top_k(logit_value, labels_to_class_names, 5)

                        check_result(fpath, path_wrong, path_low, filename, y_label, top_5, top1_p)
                        num_images += 1
                        print(y_label + '-' + str(num_images % 400) + '=', os.path.basename(filename))
                        # break
        # break
        # image_name_list.append(filename)
        # predict_labels.append(top_5)
        # predict_prob.append(top1_p)
        # print(filename, top_5)
        
    
    # np.save('./tmp/image_name_list', image_name_list)
    # np.save('./tmp/predict_labels', predict_labels)
    # np.save('./tmp/predict_prob', predict_prob)
    # print('image_name_list =', image_name_list)
    # print('predict_labels =', predict_labels)
