"""Generic evaluation script that evaluates a model using a given image.
on test data, no subdir
export top10-index.csv 
export top10-label.csv 
export all-prob.csv """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory
import numpy as np
import pandas as pd
import os
import sys


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
    allProb = list(logits[0][0])
    topChar = []
    topIndex = []
    min_label = min(predict_list)
    
    for i in range(top_k):
        label = np.argmax(predict_list)
        predict_list.remove(predict_list[label])
        predict_list.insert(label, min_label)
        topIndex.append(label)
        label_name = labels[label]
        topChar.append(label_name)
    return topChar, topIndex, allProb

# save filename , lable as csv
def save_csv(images, labels, predict, shape=(10000, 2), fname='submit_test.csv'):
    # print('shape=', shape)
    columns=['filename', 'label']
    for i in range(shape[1]):
        columns.append('col'+str(i+1))
    # print('columns=', columns)
    save_arr = np.empty((shape[0], len(columns)), dtype=np.str)
    save_arr = pd.DataFrame(save_arr, columns=columns)
    for i in range(len(images)):
        save_arr.values[i, 0] = images[i]
        save_arr.values[i, 1] = labels[i]
        for j in range(shape[1]):
            save_arr.values[i, j+2] = predict[i][j]
    save_path = os.path.join(os.getcwd(), 'tmp/'+fname)
    save_arr.to_csv(save_path, decimal=',', encoding='utf-8', index=False, index_label=False)
    print('submit_test.csv have been write, locate is :', save_path)

def run():
    with tf.Graph().as_default():
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
            print('checkpoint_path =', checkpoint_path)
            # this method is more slow then output method
            saver.restore(sess, checkpoint_path)
            
            image_name_list = []
            labels = []
            predict_labels = []
            predict_index = []
            predict_prob = []
            num_images = 0
            filenames = os.listdir(FLAGS.pic_path)

            for filename in filenames:
                # load the image
                path = os.path.join(FLAGS.pic_path, filename)
                if os.path.isdir(path):
                    labels.append(filename)
                    subFiles = os.listdir(path)
                    for subFile in subFiles:
                        path2 = os.path.join(path, subFile)
                        if os.path.isdir(path2):
                            continue
                with open(path, mode='rb') as img_file:
                    image_value = img_file.read()
                    logit_value = sess.run([logit], feed_dict={placeholder:image_value})
                    # print('logit_value = ', logit_value)
                    topChar, topIndex, allProb = get_label_predict_top_k(logit_value, labels_to_class_names, 10)
                    # print(filename, topChar)
                    # print(filename, topIndex)
                    # print(filename, allProb)

                    image_name_list.append(filename)
                    labels.append(0)
                    predict_labels.append(topChar)
                    predict_index.append(topIndex)
                    predict_prob.append(allProb)

                    num_images += 1
                    # print('num_images =', num_images)
                    sys.stdout.write('\r>> num_images %d/%d' % (num_images, len(filenames)))
                    sys.stdout.flush()
                    # break
            sys.stdout.write('\n')
            sys.stdout.flush()
            # print('image_name_list =', image_name_list)
            # print('predict_labels =', predict_labels)
            save_csv(image_name_list, labels, predict_labels, shape=(len(predict_labels), len(topChar)), fname='top10-label.csv')
            save_csv(image_name_list, labels, predict_index, shape=(len(predict_index), len(topIndex)), fname='top10-index.csv')
            save_csv(image_name_list, labels, predict_prob, shape=(len(predict_prob), len(allProb)), fname='all-prob.csv')


run()
