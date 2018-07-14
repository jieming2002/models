import os
import sys
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_group",
    type=str,
    default='train_b',
    help="the group of data: train_b, train_1w, train_c"
    )
parser.add_argument(
    "--raw_dir",
    type=str,
    default="data/raw",
    help="the path of raw data"
    )
parser.add_argument(
    "--new_dir",
    type=str,
    default="data/new",
    help="the path of new data"
    )

FLAGS, unparsed = parser.parse_known_args()
print('data_group ', FLAGS.data_group)
print('raw_dir ', FLAGS.raw_dir)
print('new_dir ', FLAGS.new_dir)

wd = os.getcwd()
sets=[(FLAGS.data_group, 'annotation')]

# classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
classes = ["vehicle"]

def convert_annotation(group, name, coordinate, list_file):
    """ convert the annotation to the format that yolo3 can read,
    then write it to list_file """
    annotation_list = coordinate.split(';')
    # print('annotation_list =', annotation_list)
    # process each annotation in 
    for annotation in annotation_list:
        cls_id = 0
        # x, y, width, height
        bbox = annotation.split('_') 
        # print('bbox =', bbox)
        if len(bbox) < 4:
            continue
        # b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        b = [int(float(bbox[0])), int(float(bbox[1])), int(float(bbox[2])), int(float(bbox[3]))]
        # print('b =', b)
        b[2] += b[0]
        b[3] += b[1]
        # print('b =', b)
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

# generate sets
for group, sub_set in sets:
    # read annotation_info
    raw_path = os.path.join(FLAGS.raw_dir, '%s.csv'%(group))
    annotation_info = pd.read_csv(raw_path)
    image_start = 0
    image_end = annotation_info.shape[0]
    print('%s range=[%s ~ %s]' % (sub_set, image_start, image_end))
    # generate list of file path
    new_path = os.path.join(FLAGS.new_dir, '%s_%s.txt'%(group, sub_set))
    with open(new_path, 'w') as list_file:
        for i in range(image_start, image_end):
            sys.stdout.write('\r>> id = %s / %s' % (i+1,image_end))
            sys.stdout.flush()
            info = annotation_info.iloc[i:i+1]
            # print('info =', info)
            name = info.name.values[0]
            # print('%s = %s' % (i, name))
            coordinate = info.coordinate.values[0]
            # print('coordinate =', coordinate)
            # skip the rows contain no target  
            if not isinstance(coordinate, str):
                print('%s = %s coordinate= %s' % (i,name,coordinate))
                continue
            if isinstance(coordinate, str) and len(coordinate) < 7:
                print('%s = %s coordinate= %s' % (i,name,coordinate))
                continue
            # write a file path to the list
            # file_path = '%s/%s/%s/%s'%(wd, FLAGS.raw_dir, group, name)
            file_path = '%s/%s' % (group, name)
            # print('file_path =', file_path)
            list_file.write(file_path)
            convert_annotation(group, name, coordinate, list_file)
            list_file.write('\n')
            # if (i > 11):
            #     break
    sys.stdout.write('\n')
    sys.stdout.flush()
