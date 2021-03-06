# **Behavioral Cloning**   

In this project for the Udacity Self-Driving Car Nanodegree a deep CNN is developed that can steer a car in a simulator provided by Udacity. The CNN drives the car autonomously around a track. The network is trained on images from a video stream that was recorded while a human was steering the car. The CNN thus clones the human driving behavior.


<div  align="center">    
<img src="image/trace1.gif" width=322 height=234 border=0/>
<img src="image/trace2.gif" width=322 height=234 border=0/>
</div>


**The goals / steps of this project are the following:**
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* **model.py:** containing the script to create and train the model
* **drive.py:** for driving the car in autonomous mode
* **model.h5:** containing a trained convolution neural network 
* **CarND-Behavioral-Cloning.md:** summarizing the results

#### 2. Submission includes functional code
Using the [Udacity self-driving car simulator](https://github.com/udacity/self-driving-car-sim) and my drive.py file, the car can be driven autonomously around the track by executing 

```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

Emphasis here, yourself data should be saved in `data` file, and the structure should like this:
```
.
├── IMG
│   ├── center_2016_12_01_13_30_48_287.jpg
│   ├── center_2016_12_01_13_30_48_404.jpg
│   ├── center_2016_12_01_13_31_12_937.jpg
│   ├── center_2016_12_01_13_31_13_037.jpg
├── my_data_1
│   └── driving_log.csv
├── my_data_2
│   └── driving_log.csv
├── my_data_3
│   ├── driving_log.csv
│   └── P3-data_3.zip
├── my_data_4
│   └── driving_log.csv
├── my_data_5
│   └── driving_log.csv
└── udacity_data
    └── driving_log.csv
```
Furthermore, you can modify this part of the program, if you don't like this structure, 
```
using_my_data = True
using_my_data_2 = True
using_my_data_3 = True
using_my_data_4 = True
using_my_data_5 = True
using_udacity_data = True
data_to_use = [using_my_data, using_my_data_2, using_my_data_3,using_my_data_4,using_my_data_5,using_udacity_data]
csv_path = ['./data/my_data_1/driving_log.csv', './data/my_data_2/driving_log.csv', './data/my_data_3/driving_log.csv', './data/my_data_4/driving_log.csv',  './data/my_data_5/driving_log.csv','./data/udacity_data/driving_log.csv']

lines = []
for j in range(len(csv_path)):
    if data_to_use[j]:
        with open(csv_path[j]) as csv_file:
            data = csv.reader(csv_file)
            for line in data:
                lines.append(line)
```
Finally, train yourself model by running 
```
python model.py
```

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is [nVidia Autonomous Car Group](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) model, and the results show that this model has a good performance (this model could be found [here](model.py#L153-L179)).

<img src="image/model.png" width="40%" height="40%" border=0/>


#### 2. Attempts to reduce overfitting in the model
In order to reduce overfitting, I use three main methods:
- **Add dropout layers:**
The model contains dropout layers in order to reduce overfitting ([model.py](model.py#L168)). 
- **Augmented data:**  
Not only use center image, but also use left and right image, and flip each image([model.py](model.py#L45))
- **Collect more data:**
In addition to the above methods, you can also use [Udacity self-driving car simulator](https://github.com/udacity/self-driving-car-sim)  to collect more data.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually ([model.py](model.py#L25)).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. Each image was used to train the model.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a simple convolution neural network model, but the test result was so bad. So, I thought use VGG16 maybe a better choice, because it generally performs well. When I was training VGG16 model, I found that each epoch was very slow for VGG16 model is too complicated.
Finally I decided to use [nVidia Autonomous Car Group](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) model.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. To combat the overfitting, I modified the model add Dropout layer.


The final step was to run the simulator to see how well the car was driving around track one. At the beginning you may encounter a car rushing out of the runway, especially when turning. In order to improve the driving behavior in these cases, You should make more pictures in the place of vehicle made a mistake.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture could be found [here](model.py#L153-L179)

A model summary is as follows:
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 90, 320, 3)        0
_________________________________________________________________
lambda_2 (Lambda)            (None, 66, 200, 3)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 31, 98, 24)        1824
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 14, 47, 36)        21636
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 5, 22, 48)         43248
_________________________________________________________________
dropout_1 (Dropout)          (None, 5, 22, 48)         0
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 3, 20, 64)         27712
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 1, 18, 64)         36928
_________________________________________________________________
dropout_2 (Dropout)          (None, 1, 18, 64)         0
_________________________________________________________________
flatten_1 (Flatten)          (None, 1152)              0
_________________________________________________________________
dense_1 (Dense)              (None, 100)               115300
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11
=================================================================
Total params: 252,219
Trainable params: 252,219
Non-trainable params: 0
_________________________________________________________________
```

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center:

<img src="image/center.jpg" width='50%' height="50%" border=0/>

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to adjust the best position. 


Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also used left、right and flipped images thinking that this would reduce overfitting. For example, here are the left, center and right images:

<div  align="center">    
<img src="image/left.jpg" width='20%' height="20%" border=0/>
Figure 1. Left
<img src="image/center.jpg" width='20%' height="20%" border=0/>
Figure 2. Center
<img src="image/right.jpg" width='20%' height="20%" border=0/>
Figure 3. Right
</div>


After the collection process, I had 153,870 number of data points. I then preprocessed this data by batch normalization、cropping and scale.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by loos result. I used an adam optimizer so that manually training the learning rate wasn't necessary. The following picture shows the training:

<div  align="center">    
<img src="image/loss.png" width='50%' height="50%" border=0/>
</div>
