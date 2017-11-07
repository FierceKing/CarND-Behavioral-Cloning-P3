"""
BATCH = 1024
BATCH_VALID = 1024
Validation dataset no shuffling
'adam' optimizer
earlystopping patience=3
"""
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Lambda, BatchNormalization
from keras import backend as K
from keras import callbacks, optimizers

import os
import sys
import argparse
from skimage.color import rgb2yuv
import math

# the commented part is for training on small dataset for code debugging
# select_data = 'capture/'
# train_records_number = 200 * 2
# valid_records_number = 51 * 2
select_data = 'data/'
train_records_number = 6428 * 2 * 3
valid_records_number = 1608 * 2 * 3
sdc_train_file = select_data + 'sdc_train.tfrecords'
sdc_valid_file = select_data + 'sdc_valid.tfrecords'
FLAGS = None
EPOCHS = 200
BATCH = 1024
BATCH_VALID = 1024
STEPS_PER_EPOCH = int(math.ceil(train_records_number / BATCH))
VALIDATION_STEPS = int(math.ceil(valid_records_number / BATCH_VALID))
SHUFFLE_BUFFER = 6428


def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
	return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def main(_):
	# path need to be modified if train on cloud hadoop system
	oss_prefix = os.path.join(FLAGS.buckets, "")

	sess = tf.Session()
	K.set_session(sess)

	def _parse_function(example_proto):
		features = {
			'measurement': tf.FixedLenFeature([], tf.float32),
			'image_raw': tf.FixedLenFeature([], tf.string)
		}
		parsed_features = tf.parse_single_example(example_proto, features)
		image = tf.image.decode_image(parsed_features["image_raw"])
		return parsed_features['measurement'], image

	# Prepare training data pipeline
	t_filenames = tf.placeholder(tf.string, shape=[None], name="train_filename")
	t_dataset = tf.contrib.data.TFRecordDataset(t_filenames)
	t_dataset = t_dataset.map(_parse_function)  # Parse the record into tensors.
	t_dataset = t_dataset.repeat()
	t_dataset = t_dataset.shuffle(buffer_size=SHUFFLE_BUFFER)
	t_dataset = t_dataset.batch(BATCH)
	iterator_train = t_dataset.make_initializable_iterator()
	train_measurement, train_image = iterator_train.get_next()

	# Prepare validation data pipeline
	v_filenames = tf.placeholder(tf.string, shape=[None], name="valid_filename")
	v_dataset = tf.contrib.data.TFRecordDataset(v_filenames)
	v_dataset = v_dataset.map(_parse_function)  # Parse the record into tensors.
	v_dataset = v_dataset.repeat()
	v_dataset = v_dataset.batch(BATCH_VALID)
	iterator_valid = v_dataset.make_initializable_iterator()
	valid_measurement, valid_image = iterator_valid.get_next()

	training_filenames = [oss_prefix + sdc_train_file]
	validation_filenames = [oss_prefix + sdc_valid_file]

	print('total records in Training file: ',
	      sum(1 for _ in tf.python_io.tf_record_iterator(training_filenames[0])))
	print('total records in Validation file:',
	      sum(1 for _ in tf.python_io.tf_record_iterator(validation_filenames[0])))

	def train_data_generator():
		sess.run(iterator_train.initializer, feed_dict={t_filenames: training_filenames})
		i = 0
		while 1:
			i += 1
			try:
				x, y = sess.run([train_image, train_measurement])
			except tf.errors.OutOfRangeError:
				break
			# convert image from RGB to YUV
			x = rgb2yuv(x)
			# crop image to focus only on the road
			x = x[:, 60:136, :, :]
			if i == 1:
				print('Input picture shape: ', x.shape)
			yield x, y

	def valid_data_generator():
		sess.run(iterator_valid.initializer, feed_dict={v_filenames: validation_filenames})
		while 1:
			try:
				x, y = sess.run([valid_image, valid_measurement])
			except tf.errors.OutOfRangeError:
				break
			x = rgb2yuv(x)
			x = x[:, 60:136, :, :]
			yield x, y

	nrows = 76
	ncols = 320

	# NVIDIA PilotNet structure
	model = Sequential()
	model.add(BatchNormalization(epsilon=0.001, axis=-1, input_shape=(nrows, ncols, 3)))
	model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
	model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
	model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
	model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
	model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
	model.add(Flatten())
	model.add(Dense(1164, activation='relu'))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(10, activation='relu'))
	model.add(Dense(1, activation='tanh'))

	model.compile(optimizer='adam', loss='mse')
	# Implement early stop mechanism
	earlystopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
	model.fit_generator(train_data_generator(),
	                    steps_per_epoch=STEPS_PER_EPOCH,
	                    epochs=EPOCHS,
	                    validation_data=valid_data_generator(),
	                    validation_steps=VALIDATION_STEPS, callbacks=[earlystopping])
	model.save('model.h5')

	# if it is training on cloud, need to save again to HDFS file system
	if oss_prefix != './':
		with open('model.h5', 'rb') as f:
			f_content = f.read()

		tf.gfile.FastGFile(oss_prefix + 'model.h5', 'wb').write(f_content)

	print("program end.")


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--buckets', type=str, default='./',
	                    help='input data path')
	parser.add_argument('--checkpointDir', type=str, default='./',
	                    help='output model path')
	FLAGS, _ = parser.parse_known_args()
	tf.app.run(main=main)
