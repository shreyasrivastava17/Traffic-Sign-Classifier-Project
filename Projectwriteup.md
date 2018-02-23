 
# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---
[//]: # (Image References)

[image1]: ./output_images/Images%20after%20Preprocessing.png
[image2]: ./output_images/Images%20before%20Processing.png 
[image3]: ./output_images/New%20Images.png 
[image4]: ./output_images/Training%20Data%20Graph.png 
[image5]: ./New_Test_Images/download.jpg  
[image6]: ./New_Test_Images/img%201.png 
[image7]: ./New_Test_Images/img%203.png 
[image8]: ./New_Test_Images/img%204.jpg
[image9]: ./New_Test_Images/img%208.jpg
[image10]: ./output_images/after%20grayscale.png
[image11]: ./output_images/before%20gray%20scale.png

**Build a Traffic Sign Recognition Project**
The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set) * Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

--
### Writeup / README
#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/shreyasrivastava17/Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)


### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.
I used the pyhton to calculate summary statistics of the traffic signs data set provided:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.
Here is an exploratory visualization of the data set. To visualize the data set I have printed 40 random image from the training data along with their label. Below is the image of 40 random images from the data set before preprocessing them.

![alt text][image2]

Here is a Bar Garph representation of the data set. It has the the clasess on the x-axis and the number of images of that class in the-axis.

![alt text][image4]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)
To preprocess the images i have performed the following the steps:
1. GrayScaling the Data: I have grayscaled the images because the colour of the traffic signs do not provide much data do the neural network while training it on the data set. I tried training the neural network on the cloured image s but tghe acuuracy was low so i chjose to garyscale the images. 

Here is an example of a traffic sign image before and after grayscaling.
Before Garscaling:

![alt text][image11]

After Garyscaling:

![alt text][image10]


2. Normalising the Data: I have normalised the data so that the data is well centered and not scattered.  
3. Shuffling the data: i have shuffled the training data so that the same kind of images are not there next to each other in the data set.

Here are some images after applying all the preprocessing steps on the data.

![alt text][image1]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The final model arcitecture that i am using has 2 convolutional layers and 3 fully connected Layes. To start with io had had given an input of 32x32x1 grayscale image to the model. The first layer is a 5x5 convolational layer with a 1x1 stride, VALID padding that gives an output of 28x28x16. On the output of the convolutional layer i have applied a RELU activaltion funtion. I have then used max pooling with a stride of 2x2 thus recieveinbg an output of 14x14x6. Then i have user a dropout on this layer with a keep_prop of 0.7 to mitigate overfitting. 
The Second layer is again a 5x5 convolutional layer that has a stride of 1x1 and valid padding. the i have used Relu as the activation function. Next i have used max pooling with a 2x2 stride and valid padding. The dropout for this layer is 30%. 
Next I have flattened the layer to produce 400 outputs. Then i have used 3 fully connected layers to ultimately give out an output of 43 classes and ahve used Relu as an activation function on all the fully connected layers.

My final model consisted of the following layers:

| Layer         		      |     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		      | 32x32x1 GaryScale image   							| 
| Convolution 5x5     	 | 1x1 stride, VALID padding, outputs 28x28x6 	|
| RELU					             |	Relu activation function											|
| Max pooling	      	   | 2x2 stride,  outputs 14x14x6 				|
| Dropout	      | Keep_prob=0.7     									|
| Convolution 5x5 	     | 1x1 stride, VALID padding, outputs 10x10x16        									|
| RELU				           | Relu activation function        									|
|	Max Pooling					| 2x2 stride, VALID padding, outputs 5x5x16												|
|	Dropout					|Keep_prob=0.7												| 
| Flatten     |Outout= 400   |
| Fully Connected Layer      |Output 120   |
|RELU|Relu activation function|
| Fully Connected Layer   |Output 84 |
|RELU| Relu activation funtion|
| Fully Connected Layer   |Output 43 |
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.


To train the model, I have use the following hyperparamerts:

| Hyper Parameter         		      |     Value	        					| 
|:---------------------:|:---------------------------------------------:| 
| Learning Rate         		      | 0.001   							| 
| Epochs     	 | 37 	|
| Batch Size					             |	128											|
| Keep Prob	      	   | 0.7 in training 				|


The model that I have trained uses a Adam Optimiser to train the model. The trained model is able to detect the Traffic signs correctly with a validation accuracy of .956

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.
My final model results were:

* training set accuracy of 96
 * validation set accuracy of .945 
* test set accuracy of .93

I have chossen a very well known architechture to start with the training of the traffic signs classifier model. The model that I have used as a base is the LeNet Architecture. I hve modified the architecture a bit to achieve this accuracy.

I beleived that the LeNet acrcitecture would be a relevent model to start with for the traffic sign application as it is complex enough. Moreover the Input to the LeNet model is 32x32x1 which is the same in this case as well as the German traffic signs dataset provided here has the images of the shape 32x32x3 and i had preprocessed the images to garyscale, thus converting them to 32x32x1.
My belief that the Lenet architecture with some changes works well for the traffic signs application as well as this is proven by the training, validation and test accuracy mentioned above. The model with very mere changes works well and gives a test accuracy of 93.6% and a validation accuracy of 95.6%. Moreover the model performs well on the unseen data, i.e it gives a accuracy of 60% on the new images that i downloaded from the internet.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

   ![alt text][image5]


   ![alt text][image6] 


   ![alt text][image7] 


   ![alt text][image8] 


   ![alt text][image9]

The Imges after preprocessing are as under:

![alt text][image3]

The first image might be difficult to classify due to the watermarks on it or due to the background noise in the pictures.

The second image might be difficult to classify due to the quality of the image.

For the third image, i think that it might not be classified correctly due to the poor quality of the image and the resemblence to the other speed signs

For the fourth image, i think that it should be clasified correctly

The fifth image might not be classified correctly due to the angling of the traffic sign in the image and the sifn mixes with the background. 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Bicycle Crossing      		| Speed Limit(80 km/hr)				| 
| Priority Road     			| Priority Road 										|
| Spped Limit(30 km/hr)					| End of Spped Limit(80 km/hr)											|
| Keep Right	      		| Keep Right				 				|
| Right-of-way at the next intersection 			| Speed Limit(60 km/hr)      							|


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. This compares favorably to the accuracy on the test set of traffic signs data

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a speed limit(60 km/hr)  sign (probability of 0.48), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .48         			| Speed Limit(80 km/hr)   									| 
| .20     				| Speed Limit(60 km/hr) 										|
| .039					| Speed Limit(120 km/hr)											|
| .035	      			| General Caution					 				|
| .034			    | Speed Limit(70 km/hr)      							|


For the second image the model guessed the image correctly as a priority road and the top 5 softmax probabilities are as below:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 3.67         			| Priority Road   									| 
| .034     				| RoundAbout Mandatory										|
| .002					| No Vehicles											|
| .0006	      			| No Passing 					 				|
| .0005      |   Road Work   |

For the third image the model guessed the image as a end of speed limit(80 km/hr) sign when it is not that sign. The top 5 softmax probabilities for the predictions are listed below.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .48         			| End of Speed Limit(80 km/hr)   									| 
| .20     				| Speed Limit(20 km/hr) 										|
| .039					| Speed Limit(60 km/hr)											|
| .035	      			| Speed Limit(80 km/hr)					 				|
| .034			    | Speed Limit(30 km/hr)      							|

For the fourth image the model was able to guess the traffic sign correctly as a Keep Right sign. The top 5 softmax probabilities are listed below.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Keep Right   									| 
| 2.97e^-16      				| Yeild 										|
| .1.16e^-17					| Children Crossing										|
| .1.63e^-18	      			| Slippery Road 				|
| .1.23e^-18			    | Right-of-way at the next intersection      							|

For the last image the model guessed the image as a Speed Limit(60 km/hr) sign but it is not a speed limit sign. The top 5 softmax probabilities are as under.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .42        			| Speed Limit(80 km/hr)   									| 
| .17     				| Speed Limit(60 km/hr) 										|
| .05					| General Caution											|
| .034	      			| Traffic Signals					 				|
| .038			    | Speed Limit(120 km/hr)      							|




