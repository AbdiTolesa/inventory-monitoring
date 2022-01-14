# Deep Learning model for Inventory Monitoring at Amazon Distribution Centers

As a leader in the industry, AWS has been making a tremendous effort to improve operational efficiency in its Distribution Centers. The increasingly usage of robots to move objects for fulfillment of customer orders is one of those moves acted towards that. Robots are used to move objects into bins. This project is about building a model that can count the number of objects in each bin. A system like this can be used to track inventory and make sure that delivery consignments have the correct number of items.


## Project Set Up and Installation
This project requires setting up Amazon SageMaker studio. If you have not done that and not sure how to do it, please refer to [this link](https://docs.aws.amazon.com/sagemaker/latest/dg/onboard-quick-start.html)

## Dataset

### Overview
The Amazon Bin Image Dataset contains images and metadata from bins of a pod in an operating Amazon Fulfillment Center. The bin images in this dataset are captured as robot units carry pods as part of normal Amazon Fulfillment Center operations. You can download and find the details at [here](https://aws.amazon.com/ko/public-datasets/amazon-bin-images/). The dataset has over 500,000 bin JPEG images and corresponding JSON metadata files describing items in bins in Amazon Fulfillment Centers.

### Access
The data is downloaded from the source and then rearranged to make it convenient for the Deep Learning algorithm in such a way that the images are placed in their respective directory name that represents the number of objects in the bins. The function `download_and_arrange_data()` does that by using `file_list.json` file, which contains a subset of file names used to train our model, since the original dataset is very large. There is a function written to achieve that. A script is also written to split the dataset into train and validation sets with the recommended 80%-20% ratio.
AWS CLI is then used to upload the data to AWS S3 bucket.

## Model Training
What kind of model did you choose for this experiment and why? Give an overview of the types of hyperparameters that you specified and why you chose them. Also remember to evaluate the performance of your model. 

A pretrained version of ResNet50 is used to build the model to save time than training the model from scratch. ResNet50 is 50 layers deep CNN and used commonly to solve similar problems. 

SageMaker's [hyperparameter tuning job](https://sagemaker.readthedocs.io/en/stable/api/training/tuner.html) is used to pick the best hyperparameters to train our algorithm from a range of values given. The hyperparameters specified are the number of epochs, batch-size and learning-rate as they are commonly used and affect the model's performance at large.

[PyTorch with the SageMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html#save-the-model) is used for implementing training the algorithm.

## Inference
The model takes in an image and then gives an output of an integer that is a prediction of the number of objects in the bin.

## Machine Learning Pipeline
**TODO:** Explain your project pipeline.
To explain about the ML Pipeline used in this project, first of all, the dataset is downloaded and arranged based on a json file containing the partial list of files used to train the model. This is just for the save of saving time and cost as the original dataset is very huge, 500,000 images. The data is then uploaded to AWS S3 bucket to be consumed by the training script later in the training phase. A PyTorch estimator is created in SageMaker studio that infers the training script's location as one of its parameters along with the hyperparameters to be used. The performance that is achieved in the setup and options described would yield an accuracy of 32%. Although this is a very low performance, it is so because of the size of the subset of data used.

## Standout Suggestions
**TODO (Optional):** This is where you can provide information about any standout suggestions that you have attempted.
