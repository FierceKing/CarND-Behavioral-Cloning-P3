"""
Do data augmentations including mirroring, and use 3 cameras
Save the data into tfrecord format, for higher performance
"""
import tensorflow as tf
import numpy as np
import csv

import os
import sys
import argparse

from PIL import Image, ImageOps
from io import BytesIO

FLAGS = None
data_folder = 'data/'
# data_folder = 'capture/'
train_filename = data_folder + 'sdc_train.tfrecords'
validation_file = data_folder + 'sdc_valid.tfrecords'
csv_file_name = data_folder + 'driving_log.csv'
image_folder = data_folder + 'IMG/'


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def main(_):
    oss_prefix = os.path.join(FLAGS.buckets, "")
    if oss_prefix != './':
        oss_prefix = 'cached://' + oss_prefix

    lines = []
    with tf.gfile.FastGFile(oss_prefix + csv_file_name, mode='r') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
        print('load file names finished')
    # shuffle file path
    np.random.shuffle(lines)

    # split to training and validation set
    separator = int(len(lines) * 0.8)
    lines_train = lines[:separator]
    lines_valid = lines[separator:]

    def write_to_file(img, measure, writer):
        example = tf.train.Example(features=tf.train.Features(feature={
            'measurement': _float_feature(measure),
            'image_raw': _bytes_feature(img)}))
        writer.write(example.SerializeToString())
        return

    def augment(lines_input, filename):
        """
        Augmentation of data
        1. flip all images to let the model learn to turn in different directions
        2. for left/right steering, +/- 1 deg
        3. for flipped image's steering, multiply by -1
        :param lines_input: file path list, which contains 3 paths on each line: center,left,right
        :param filename: the filename for tfrecords file
        :return: no return
        """
        tfrecord_file = oss_prefix + filename
        writer = tf.python_io.TFRecordWriter(tfrecord_file)
        print("{} records in {}".format(len(lines_input), filename))
        count_line = 0
        count_total = 0
        for line in lines_input:
            print("count line {}, total {}".format(count_line, count_total))
            count_line += 1
            for i in range(3):
                count_total += 1
                source_path = line[i]
                img_filename = source_path.split('/')[-1]
                current_path = oss_prefix + image_folder + img_filename

                image_raw = Image.open(current_path)
                image_original = image_raw
                augmented_image_raw = ImageOps.mirror(image_original)
                image_raw = open(current_path, 'rb').read()

                f2 = BytesIO()
                augmented_image_raw.save(f2, 'PNG')
                augmented_image_raw = f2.getvalue()
                if i == 0:
                    measurement = float(line[3])
                elif i == 1:
                    measurement = float(line[3]) + 1.0 / 25.0
                elif i == 2:
                    measurement = float(line[3]) - 1.0 / 25.0
                augmented_measurement = measurement * -1.0

                write_to_file(image_raw, measurement, writer)
                write_to_file(augmented_image_raw, augmented_measurement, writer)
        writer.close()
        print("finish writing: {}".format(tfrecord_file))
        return

    augment(lines_train, train_filename)
    augment(lines_valid, validation_file)


# In case to do the job on cloud, need to save to HDFS
# def save_to_oss(filename):
# 	with open(filename, 'rb') as f:
# 		tf.gfile.FastGFile(oss_prefix + filename, 'wb').write(f.read())
#
# if oss_prefix != './':
# 	save_to_oss(train_filename)
# 	save_to_oss(validation_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--buckets', type=str, default='./',
                        help='input data path')
    parser.add_argument('--checkpointDir', type=str, default='./',
                        help='output model path')
    FLAGS, _ = parser.parse_known_args()
    tf.app.run(main=main)
