import argparse
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image

from utils import visualization_utils as vis_util
from utils import label_map_util

if tf.__version__ < '1.4.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

NUM_CLASSES = 1


def parse_args(check=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--images_dir', type=str, required=True)
    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS, unparsed


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def show_result_on_image(image_np, boxes, classes, scores):
    image_np = vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        min_score_thresh=0.2,
        line_thickness=8)
    plt.imsave(os.path.join(FLAGS.output_dir, 'output.png'), image_np)

def convert_result_to_str(boxes, scores, classes, image, min_score_thresh=.2):
    out_str = ''
    out_str_int = ''
    for i, score in reversed(list(enumerate(scores[0]))):
        box = boxes[0][i]
        if score > min_score_thresh:
            # print('i=%s, score=%s box=%s' % (i, score, box))
            top, left, bottom, right = box

            top = max(0, top*image.size[1])
            left = max(0, left*image.size[0])
            bottom = min(image.size[1], bottom*image.size[1])
            right = min(image.size[0], right*image.size[0])
            width = right - left
            height = bottom - top

            top_int = np.round(top).astype('int32')
            left_int = np.round(left).astype('int32')
            width_int = np.round(width).astype('int32')
            height_int = np.round(height).astype('int32')

            # print(i, (left, top), (right, bottom), (width, height))
            out_str += '_'.join([str(left), str(top), str(width), str(height)])
            out_str_int += '_'.join([str(left_int), str(top_int), str(width_int), str(height_int)])
            if i > 0:
                out_str_int += ';'
                out_str += ';'
    return out_str_int, out_str


def test_images_in_dir(image_tensor, detection_boxes, detection_scores, detection_classes, num_detections, sess):
    names = []
    boxes_str = []
    boxes_str_int = []
    file_list = os.listdir(FLAGS.images_dir)
    total = len(file_list)
    
    for i, filename in enumerate(file_list):
        path = os.path.join(FLAGS.images_dir, filename)
        if not os.path.isdir(path):
            try:
                image = Image.open(path)
            except:
                print('Open Error! Try again! %s' % path)
                continue
            else:
                image_np = load_image_into_numpy_array(image)
                image_np_expanded = np.expand_dims(image_np, axis=0)
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                # print('skye boxes=', boxes)
                # print('skye scores=',scores)
                # print('skye classes=', classes)
                # show_result_on_image(image_np, boxes, classes, scores)

                names.append(filename)
                # print('filename=', filename)
                out_str_int, out_str = convert_result_to_str(boxes, scores, classes, image)
                # print('out_str_int=', out_str_int)
                # print('out_str=', out_str)
                boxes_str_int.append(out_str_int)
                boxes_str.append(out_str)
                sys.stdout.write('\r>> i = %s / %s' % ((i+1),total))
                sys.stdout.flush()
        # break
    sys.stdout.write('\n')
    sys.stdout.flush()

    summit = pd.DataFrame({'name':names})
    summit['coordinate'] = boxes_str_int
    path = os.path.join(FLAGS.output_dir, 'summit_int.csv')
    summit.to_csv(path, index=False)
    print('path =', path)

    summit = pd.DataFrame({'name':names})
    summit['coordinate'] = boxes_str
    path = os.path.join(FLAGS.output_dir, 'summit.csv')
    summit.to_csv(path, index=False)
    print('path =', path)


if __name__ == '__main__':
    FLAGS, unparsed = parse_args()

    PATH_TO_CKPT = os.path.join(FLAGS.output_dir, 'frozen_inference_graph.pb')
    PATH_TO_LABELS = os.path.join(FLAGS.dataset_dir, 'labels_items.txt')

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            test_images_in_dir(image_tensor, detection_boxes, detection_scores, detection_classes, num_detections, sess)

