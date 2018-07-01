# 图像数据增广 Deep learning image augmentation

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf  
import numpy as np 
import os
import matplotlib.pyplot as plt

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    'dataset_dir',
    None,
    'The directory where the images.')


def distort_color(image, color_ordering=0):  
    if color_ordering == 0:  
        image = tf.image.random_brightness(image, max_delta=32. / 255.)#亮度  
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)#饱和度  
        image = tf.image.random_hue(image, max_delta=0.2)#色相  
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)#对比度  
    if color_ordering == 1:  
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)  
        image = tf.image.random_hue(image, max_delta=0.2)  
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)  
        image = tf.image.random_brightness(image, max_delta=32. / 255.)  
    if color_ordering == 2:  
        image = tf.image.random_hue(image, max_delta=0.2)  
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)  
        image = tf.image.random_brightness(image, max_delta=32. / 255.)  
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)  
    if color_ordering == 3:  
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)  
        image = tf.image.random_brightness(image, max_delta=32. / 255.)  
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)  
        image = tf.image.random_hue(image, max_delta=0.2)  
    return tf.clip_by_value(image, 0.0, 1.0)  


def preprocess_for_train(image, height, width, bbox):  
    if bbox is None:  
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])  
    if image.dtype != tf.float32:  
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)  
    
    # bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(tf.shape(image), bounding_boxes=bbox, min_object_covered=0.1)  
    # distorted_image = tf.slice(image, bbox_begin, bbox_size)  
    # distorted_image = tf.image.resize_images(distorted_image, [height, width], method=np.random.randint(4))  

    distorted_image = tf.image.resize_images(image, [height, width], method=np.random.randint(4))  
    distorted_image = tf.image.random_flip_left_right(distorted_image)  
    distorted_image = distort_color(distorted_image, np.random.randint(4))  
    return distorted_image  


def _get_filenames_and_classes(dataset_dir):
    """Returns a list of filenames
    """
    flower_root = os.path.join(dataset_dir, 'train')
    augment_root = os.path.join(dataset_dir, 'train_augment')
    if not tf.gfile.Exists(augment_root):
        tf.gfile.MakeDirs(augment_root)
    
    photo_filenames = []
    for filename in os.listdir(flower_root): #根目录
        path = os.path.join(flower_root, filename)
        # print('path =', path)
        if os.path.isdir(path): #子目录
            augment_path = os.path.join(augment_root, filename)
        if not tf.gfile.Exists(augment_path):
            tf.gfile.MakeDirs(augment_path)
        for filename in os.listdir(path): #遍历子目录 
            fpath = os.path.join(path, filename)
            photo_filenames.append(fpath)
    return photo_filenames


def preprocess_each(filenames):
    boxes = tf.constant([[[0.0, 0.0, 1.0, 1.0], [0.1, 0.1, 0.9, 0.9]]])
    fnum = 0
    for fname in filenames:
        if fnum % 10 == 0:
            print(fnum, fname)
        fnum = fnum + 1
        value = tf.gfile.FastGFile(fname, 'rb').read()
        image_data = tf.image.decode_jpeg(value, channels=3) 
        #对一个图片进行多次增广 1~9 共 9 次，加上原图，训练集总共扩大到原来的 10 倍
        for i in range(1, 2): 
            result = preprocess_for_train(image_data, 299, 299, boxes)
            yield (i, fname, result)


def get_outputname(filename, ix):
    f_dir = os.path.dirname(filename)
#     print(f_dir)
    dir_name = os.path.basename(f_dir)
#     print(dir_name)
    p_dir = os.path.dirname(os.path.dirname(f_dir))
#     print(p_dir)
    basename = os.path.basename(filename)
#     print(basename)
    output_filename = '%s/train_augment/%s/augment%d_%s' % (p_dir, dir_name, ix, basename)
#     print(output_filename)
    return output_filename


print('dataset_dir =', FLAGS.dataset_dir)
#读取图像可任意大小  
filenames = _get_filenames_and_classes(FLAGS.dataset_dir)
# global_step = 0
step_size = 1000 #每次 1000 个 
step = 1 #共 40 次 

with tf.Session() as sess:  
#     coord = tf.train.Coordinator()  
#     threads = tf.train.start_queue_runners(coord=coord)  
  
    init = tf.global_variables_initializer()  
    sess.run(init)  
    
    for ix, fname, result in preprocess_each(filenames[(step-1)*step_size:step*step_size+1]): 
        output_filename = get_outputname(fname, ix)
#         global_step = global_step + 1
#         if global_step % 9 == 0:
#             print(global_step, output_filename)
        reimg = result.eval()  
        plt.imsave(output_filename, reimg)  #这个速度快 
  
#     coord.request_stop()  
#     coord.join(threads)  

