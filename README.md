#**Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* `convert_to_TFRecord.py` for converting images into augmented data in TFRecords format
* `learn_tfrecords_final.py` containing the script to create and train the model
* `drive.py` for driving the car in autonomous mode
* `model.h5` containing a trained convolution neural network, can be download from this repository or [here - at Aliyun OSS](http://udacitycarnd.oss-cn-shanghai.aliyuncs.com/github_link_file/behavior_cloning/model.h5).
* `README.md` summarizing the results
* `self-driving.mov` video recorded during the test run of `model.h5` on the first track. it can be downloaded [here - at Aliyun OSS](http://udacitycarnd.oss-cn-shanghai.aliyuncs.com/github_link_file/behavior_cloning/self-driving.mov) or watch [Youtube](https://youtu.be/jwCmUppWUos).

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```
python drive.py model.h5
```

####3. Submission code is usable and readable

To train the network using my `learn_tfrecords_final.py`, first you need to convert the data captured from Udacity self driving car simulator in to `TFRecord` format. It also augment the data by adding left/right camera data, and mirroring the images, the steering will also be modified according to some simple assumptions, will disscuss in later part.

The `learn_tfrecords_final.py` file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a NVIDIA PilotNet architecture. There is a batch normalizaiton layer at the bottom, following 5 convolution layers, to which an relu activation will be applied for non-linearty, than flatten the features and feed into 5 fully connected layers, on the top is one neuron predicting the steering ratio.

####2. Attempts to reduce overfitting in the model

The model doesn't contain dropout as it is what NVIDIA did, but the validation result is monitord by early stopping mechanism using Keras callback functionality.

```ruby
earlystopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
```

In the `convert_to_TFRecord.py` the acquiered data is random shuffled and then spilt into training/validation dataset in a 80/20 ratio.The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 140).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, add/deduce 1 degree from the steering to simulate the image is taken from the center camera, thus need to turn 1 degree to get back to the middle of the lane.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to use a proved strong architecture instead of build from scratch.

My first step was to use a convolution neural network model similar to the NVIDIA PilotNet I thought this model might be appropriate because it is the same scenario. But it might be too big for my data, because my data is simpler and smaller, if overfitting is observed, I could use some approch to mitigate.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that if the mean squared error on the training data will soon reduce below 0.1, than reduce slowly but continue going down, while the validation error will stop decreasing once it reach certain level and start to fluctuate. This implied that the model was overfitting. Since the model is capable of doing more complex task according to NVIDIA, I didn't try to apply dropout, in order to train the model better once I get more training data, eg. `Track 2` .

But over fitting cannot be ignored, so I implemented early stopping mechanism, once the validation loss stop decressing for 3 epochs, it will finish training.

In the spirit of curiosity, I want to see if doing something else can make the model better, I tried to added batch normalizaiton after every convolution layers, it didn't slow down the training epochs significantly, but the training and validation error becomes very different, and the validation error cannot be lowerd to a plesent value, shown as below.

```
loss: 0.0086 - val_loss: 0.0448
```
The early stopping mechanism shut it down in the 6th epoch, in contrast of the good model, it would train around 20 epochs, the loss would be something like,

```
loss: 0.0049 - val_loss: 0.0097
```

The final step was to run the simulator to see how well the car was driving around track one. When I used very small amount of data and very simple model (eg. only 1 fully connected layer), when I tested it, I wouldn't say it is actually driving, looks like no driver at all. When I say the NVIDIA model use YUV instead of RGB, I checked on the internet, it is said that YUV is more fit for human vision, although the neural network can somehow learn to convert RGB to another space which it would understand better. Details can be found here:

[Merging chrominance and luminance in early, medium and late fusion using Convolutional Neural Networks](http://users.ics.aalto.fi/perellm1/thesis.shtml)

Why not we train it like human! So before the data is fed into the model, I convert the image into YUV color space. It worked! the model can make the first turn already, note that the data I used to test training only consist of about 10 seconds images! later data augmentation is added to the full loop dataset, and before the image is fed into the model, it is cropped to 76*320, will discuss later.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (`learn_tfrecords_final.py` lines 126-138) consisted of a convolution neural network with the following layers and layer sizes. It's actually almost the same like NVIDIA PilotNet. The input shape is (76, 32, 3) and in YUV color space.

```ruby
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
```

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I woundn't want to spend too much time on practicing driving the simulater, so I used data provided by udacity.Here is an example image of center lane driving:

![](writeup_pictures/center_2016_12_01_13_30_48_287.jpg)

There are also pictures recorded from the left and right side. It could be used to simulate driving recovery form off center, only need to +/- 1 degree, this number comes from eyeballing, because the real algrithm is complicated and would not be much better from a human driver's experience (I have 7 years of driving experience).

To augment the dataset further, I also flipped images and angles thinking that this would generalize for the turns, as the 1st track contains mostly left turns, if not add flipped images, the model will have a tendency to turn left. 

After the collection process, I had 48216 number of data points. Before the data is feed into Keras model, I convert the image to YUV color space, and crop to size of (76, 320, 3), it is not incoporated to the dataset, because when testing with the simulater, I would get RGB images and not cropped. The `drive.py` file is also modified to have the capability of converting color space and image cropping.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

All the data is prepared with `convert_to_TFRecord.py`, augmentation steps are as follow:

	1. flip all images to let the model learn to turn in different directions
	2. for left/right steering, +/- 1 deg
	3. for flipped image's steering, multiply by -1

Before the images are fed into Keras model, they are further processed by:
	
	1. conver from RGB to YUV color space
	2. cropping to (76, 320, 3) 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by early stopping mechanism, it is realized with Keras call back method, with patience of 3, which means the model will stop training if the validation loss stop decressing for 3 epochs. I used an adam optimizer so that manually training the learning rate wasn't necessary.
