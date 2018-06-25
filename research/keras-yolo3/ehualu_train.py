"""
Retrain the YOLO model for your own dataset.
new feature: support multi dataset, each dataset = {group}.zip、{group}_annotation.txt. 
"""

import numpy as np
import argparse
import os
import zipfile

import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data


def _main():
    groups = FLAGS.groups.split(',')
    log_dir = FLAGS.output_dir
    classes_path = FLAGS.classes_path
    anchors_path = FLAGS.anchors_path
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)
    # print('num_classes=%s class_names=%s' % (num_classes,class_names))

    input_shape = (416,416) # multiple of 32, hw

    is_tiny_version = len(anchors)==6 # default setting
    if is_tiny_version:
        model = create_tiny_model(input_shape, anchors, num_classes,
            freeze_body=2, weights_path=FLAGS.tiny_weights_path)
    else:
        model = create_model(input_shape, anchors, num_classes,
            freeze_body=2, weights_path=FLAGS.weights_path) # make sure you know what you freeze

    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    
    lines, num_train, num_val = get_annotations(groups)
    zip_dict = get_zip_dict(groups)

    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if True:
        model.compile(optimizer=Adam(lr=FLAGS.learning_rate), loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred})

        batch_size = FLAGS.batch_size
        epoch_1 = FLAGS.epoch #50
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(
            data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes, zip_dict=zip_dict),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes, zip_dict=zip_dict),
            validation_steps=max(1, num_val//batch_size),
            epochs=epoch_1, 
            initial_epoch=0,
            callbacks=[logging, checkpoint])
        model.save_weights(os.path.join(log_dir,'trained_weights_stage_1.h5'))

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        # recompile to apply the change
        model.compile(optimizer=Adam(lr=FLAGS.learning_rate * 0.1), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) 
        print('Unfreeze all of the layers.')

        batch_size = FLAGS.batch_size_2 # note that more GPU memory is required after unfreezing the body
        epoch_2 = FLAGS.epoch_2 #100
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes, zip_dict=zip_dict),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes, zip_dict=zip_dict),
            validation_steps=max(1, num_val//batch_size),
            epochs=epoch_2, 
            initial_epoch=epoch_1,
            callbacks=[logging, checkpoint, reduce_lr, early_stopping])
        model.save_weights(os.path.join(log_dir,'trained_weights_final.h5'))

    # Further training if needed.
    close_zip_dict(zip_dict, groups)
    print('training complete!')


def get_annotations(groups):
    lines = []
    for group in groups:
        path = os.path.join(FLAGS.dataset_path, '%s_annotation.txt' % group)
        with open(path) as f:
            lines += f.readlines()
            # print('lines=', len(lines))
    
    val_split = 0.1
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    return lines, num_train, num_val


def get_zip_dict(groups):
    zip_dict = {}
    for group in groups:
        zip_path = os.path.join(FLAGS.dataset_path, '%s.zip' % group)
        zip_dict[group] = zipfile.ZipFile(zip_path, 'r')
    return zip_dict


def close_zip_dict(zip_dict, groups):
    for group in groups:
        zfile = zip_dict[group]
        zfile.close()


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/yolo_weights.h5'):
    '''create the training model'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
        num_anchors//3, num_classes+5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    load_pretrained = False
    if load_pretrained:
        # model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        model_body.load_weights(weights_path, by_name=True) # for old version
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers)-3)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model

def create_tiny_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/tiny_yolo_weights.h5'):
    '''create the training model, for Tiny YOLOv3'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16}[l], w//{0:32, 1:16}[l], \
        num_anchors//2, num_classes+5)) for l in range(2)]

    model_body = tiny_yolo_body(image_input, num_anchors//2, num_classes)
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    load_pretrained = False
    if load_pretrained:
        # model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        model_body.load_weights(weights_path, by_name=True) # for old version
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze the darknet body or freeze all but 2 output layers.
            num = (20, len(model_body.layers)-2)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model

def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes, zip_dict=None):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=True, zip_dict=zip_dict)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)

def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes, zip_dict=None):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes, zip_dict=zip_dict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        default='data/new/',
        help="dataset_path"
        )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/logs/ehualu_000",
        help="output_dir"
        )
    parser.add_argument(
        "--classes_path",
        type=str,
        default="model_data/ehualu_classes.txt",
        help="the path of classes file"
        )
    parser.add_argument(
        "--anchors_path",
        type=str,
        default="model_data/yolo_anchors.txt",
        help="the path of anchors file"
        )
    parser.add_argument(
        "--tiny_weights_path",
        type=str,
        default="model_data/tiny_yolo_weights.h5",
        help="the path of tiny_weights file"
        )
    parser.add_argument(
        "--weights_path",
        type=str,
        default="model_data/ehualu_yolo_weights.h5",
        help="the path of weights file"
        )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="the batch_size of training frozen layers "
        )
    parser.add_argument(
        "--epoch",
        type=int,
        default=1,
        help="the epoch num of training frozen layers  "
        )
    parser.add_argument(
        "--batch_size_2",
        type=int,
        default=1,
        help="the batch_size of training all layers, smaller then batch_size_1, because need more resouce."
        )
    parser.add_argument(
        "--epoch_2",
        type=int,
        default=2,
        help="the epoch num of training all layers, should big then epoch_1."
        )
    parser.add_argument(
        "--groups",
        type=str,
        default='a,b,c',
        help="the groups of dataset."
        )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="the learning_rate."
        )

    FLAGS, unparsed = parser.parse_known_args()
    print('dataset_path =', FLAGS.dataset_path)
    print('output_dir =', FLAGS.output_dir)
    print('classes_path =', FLAGS.classes_path)
    print('anchors_path =', FLAGS.anchors_path)
    print('tiny_weights_path =', FLAGS.tiny_weights_path)
    print('weights_path =', FLAGS.weights_path)
    print('batch_size =', FLAGS.batch_size)
    print('epoch =', FLAGS.epoch)
    print('batch_size_2 =', FLAGS.batch_size_2)
    print('epoch_2 =', FLAGS.epoch_2)
    print('groups =', FLAGS.groups)
    print('learning_rate =', FLAGS.learning_rate)

    _main()
