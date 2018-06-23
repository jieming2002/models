#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run a YOLO_v3 style detection model on test images.
"""

import colorsys
import os
import argparse
from timeit import default_timer as timer

import numpy as np
import pandas as pd
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image


class YOLO(object):
    def __init__(self, model_path, classes_path, anchors_path):
        self.model_path = model_path
        self.anchors_path = anchors_path
        self.classes_path = classes_path

        self.score = 0.3
        self.iou = 0.45
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.model_image_size = (416, 416) # fixed size or (None, None), hw
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        # print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        # print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        # self.draw_result_on_img(out_boxes, out_scores, out_classes, image)
        out_str = self.convert_result_to_str(out_boxes, out_scores, out_classes, image)

        end = timer()
        # print('time =',end - start)
        return image, out_str
    
    
    def convert_result_to_str(self, boxes, scores, classes, image):
        out_str = ''
        # from len to 0
        for i, c in reversed(list(enumerate(classes))):
            box = boxes[i]
            score = scores[i]
            # print('box =', box)
            
            top, left, bottom, right = box
            top = max(0, np.round(top).astype('int32'))
            left = max(0, np.round(left).astype('int32'))
            bottom = min(image.size[1], np.round(bottom).astype('int32'))
            right = min(image.size[0], np.round(right).astype('int32'))
            width = right - left
            height = bottom - top
            
            # print(i, (left, top), (right, bottom), (width, height))
            # out_str += [top,left,width,height].join('_')
            out_str += '_'.join([str(left), str(top), str(width), str(height)])
            if i > 0:
                out_str += ';'
        return out_str
    
    
    def draw_result_on_img(self, boxes, scores, classes, image):
        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(classes))):
            predicted_class = self.class_names[c]
            box = boxes[i]
            score = scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

    def close_session(self):
        self.sess.close()


def detect_img(yolo):
    img_root = FLAGS.img_root
    names = []
    boxes_str = []
    file_list = os.listdir(img_root)
    total = len(file_list)

    for i, filename in enumerate(file_list):
        path = os.path.join(img_root, filename)
        if not os.path.isdir(path):
            try:
                image = Image.open(path)
            except:
                print('Open Error! Try again! %s' % path)
                continue
            else:
                r_image, out_str = yolo.detect_image(image)
                # r_image.show()
                # print('out_str =', filename+' '+out_str)
                names.append(filename)
                boxes_str.append(out_str)
                print('i =', (i,total))
        # break
    summit = pd.DataFrame({'name':names})
    summit['coordinate'] = boxes_str
    path = os.path.join(FLAGS.summit_dir, 'summit.csv')
    summit.to_csv(path, index=False)

    yolo.close_session()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default='model_data/yolo.h5',
        help="model path or trained weights path"
        )
    parser.add_argument(
        "--classes_path",
        type=str,
        default="model_data/coco_classes.txt",
        help="classes file path"
        )
    parser.add_argument(
        "--anchors_path",
        type=str,
        default="model_data/yolo_anchors.txt",
        help="anchors file path"
        )
    parser.add_argument(
        "--img_root",
        type=str,
        default="",
        help="the root of images"
        )
    parser.add_argument(
        "--summit_dir",
        type=str,
        default="",
        help="the summit_dir"
        )

    FLAGS, unparsed = parser.parse_known_args()
    print('model_path ', FLAGS.model_path)
    print('classes_path ', FLAGS.classes_path)
    print('img_root ', FLAGS.img_root)
    print('summit_dir ', FLAGS.summit_dir)

    detect_img(YOLO(FLAGS.model_path, FLAGS.classes_path, FLAGS.anchors_path))

