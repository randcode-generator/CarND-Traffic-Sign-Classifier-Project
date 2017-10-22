# **Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[explore1]: ./img/explore1.png "Random Image"
[explore2]: ./img/explore2.png "Set of Images"
[process1]: ./img/process1.png "Before Equalization"
[process2]: ./img/process2.png "After Equalization"
[sign1]: ./mysigns/sign1.jpg ""
[sign2]: ./mysigns/sign2.jpg ""
[sign3]: ./mysigns/sign3.jpg ""
[sign4]: ./mysigns/sign4.jpg ""
[sign5]: ./mysigns/sign5.jpg ""
[sign1softmax]: ./img/sign1softmax.png
[sign2softmax]: ./img/sign2softmax.png
[sign3softmax]: ./img/sign3softmax.png
[sign4softmax]: ./img/sign4softmax.png
[sign5softmax]: ./img/sign5softmax.png

Link to [final notebook](Traffic_Sign_Classifier.html)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set.

* Number of training examples = 34799
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43

#### 2. Include an exploratory visualization of the dataset.

* Random image

![alt text][explore1]

First pass, retrieve random image. This code was taken from another example.

* Set of images

![alt text][explore2]

Second pass, filter array based on class (sign type). Show 9 images at a time.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data.

I used a technique called **Histogram equalization**. It adjusts the contrasts and converts the image range from 0-255 to 0-1.

* Before

![alt text][process1]

* After

![alt text][process2]

Finally, I normalize the data by subtracting 0.5 and dividing by 0.5 from all (RGB) values in the image. 

#### 2. Describe your final model architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					| Zero out negative values						|
| Max pooling	      	| 2x2 stride, valid padding, outputs 14x14x6 	|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					| Zero out negative values						|
| Max pooling	      	| 2x2 stride, valid padding, outputs 5x5x16 	|
| Flatten               | Flattens a 5x5x16 array to 400 elements		|
| Fully connected		| 400x200 weights, 200 bias 					|
| Sigmoid				| Self-explanatory								|
| Fully connected		| 200x80 weights, 80 bias						|
| Sigmoid				| Self-explanatory								|
| Logits				| 80x43 weights, 43 bias 


#### 3. Describe how you trained your model

1. Do a softmax with cross entropy between the logit (output of the LeNet model) and the actual class.
2. Minimize the mean of the softmax using Adam Optimizer with learning rate of 0.005
3. The number of epochs is 10 and batch size is 128

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93.

My final model results were:
* training set accuracy of 0.998
* validation set accuracy of 0.943 
* test set accuracy of 0.924

I use the LeNet architecture because it worked well with classifying written numbers. The initial architecture didn't give the 0.93 required. 

The following steps increased accuracy:
1. Increasing the number of layers for fully connected
2. Changing the activation function to sigmoid for fully connected 

What I found out:
1. Higher number of fully connected layers will result in overfitting. It will learn the training data.
2. Lower number of full connected layers will result in underfitting. It won't perform well even on clean images with no background and good contrast.
3. Using dropouts did not improve accuracy.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][sign1] ![alt text][sign2] ![alt text][sign3] 
![alt text][sign4] ![alt text][sign5]

The following might confuse the algorithm:
1. The first 2 signs has a gray background
2. The third signs has leaves in the background
3. The last 2 signs has transparent background

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set
Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (70km/h)  | Speed limit (70km/h)  						| 
| Speed limit (50km/h)  | Speed limit (50km/h)							|
| Stop					| Stop											|
| Ahead only	      	| Ahead only					 				|
| Go straight or right	| Go straight or right      					|


Amazingly, the model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

**Sign 1**

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.986         		| Speed limit (70km/h)   						| 
| 0.005     			| Speed limit (20km/h) 							|
| 0.004					| Speed limit (120km/h)							|
| 0.003	      			| Speed limit (30km/h)					 		|
| 0.001				    | General caution     							|

![alt text][sign1softmax]

The neural network is highly certain that the sign is a **speed limit at 70km/h** sign

**Sign 2**

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.869 				| Speed limit (50km/h)							|
| 0.099 				| Speed limit (80km/h)							|
| 0.012 				| Speed limit (60km/h)							|
| 0.010 				| Speed limit (30km/h)							|
| 0.004 				| Speed limit (20km/h)							|

![alt text][sign2softmax]

The neural network is highly certain that the sign is a **speed limit at 50km/h** sign

**Sign 3**

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.987					| Stop											|
| 0.008					| No entry 										|
| 0.002					| Road work 									|
| 0.001					| Speed limit (60km/h) 							|
| 0.001					| Speed limit (80km/h) 							|

![alt text][sign3softmax]

The neural network is highly certain that the sign is a **stop** sign

**Sign 4**

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.943					| Ahead only 									|
| 0.010					| Speed limit (60km/h) 							|
| 0.006					| Dangerous curve to the left 					|
| 0.005					| Turn right ahead 								|
| 0.005					| End of no passing 							|

![alt text][sign4softmax]

The neural network is highly certain that the sign is a **ahead only** sign

**Sign 5**

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.400					| Go straight or right 							|
| 0.201					| Priority road 								|
| 0.137					| Keep right 									|
| 0.054					| Yield 										|
| 0.049					| End of no passing 							|

![alt text][sign5softmax]

Unlike the previous signs, the probability for **go straight or right** sign is low and the probability for **priority road** sign is not too far from the **go straight or right** sign

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

The first convolution turned a RGB image into a 28x28 grayscale image. The convolution acted like an edge detector. The RELU turned the light (or gray) pixel into dark pixels. The MaxPool chooses the largest value which are the lightest pixels.

It looks like the neural network will use the lightest pixels as its features.