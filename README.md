# Description

In this part, I work with a set of Tweets about US airlines and examine their sentiment polarity. More details about the dataset can be found on the website [Kaggle](https://www.kaggle.com/crowdflower/twitter-airline-sentiment). My aim is to learn to classify Tweets as either "positive", "neutral", or "negative" by using two classifiers and pipelines for pre-processing and model building.

All this has to be done in a Scala class, which has to be part of a Scala SBT or Maven project. Make sure you have all your dependencies and the class can be run on AWS. The class will have 2 parameters one that represents the path of the input file and the second one that represents the output path where the output will be stored.


# Procedure

Below are the steps of the project:

1. **Loading**: First step is to define an input argument that defines the path from which to load the dataset. After that, you will need to remove rows where the text field is null.
2. **Pre-Processing**: You will start by creating a pre-processing pipeline with the following stages:
	- **Tokenizer:** Transform the text column by breaking down the sentence into words
	- **Stop Word Remover:** Remove stop-words from the words column
	- **Term Hashing:** Convert words to term-frequency vectors
	- **Label Conversion:** The label is a string e.g. \Positive", which you need to convert to numeric format
3. **Model Creation:** Logistic Regression and SVM are used with parameter chosen from 5-fold Cross Validation.

# Evaluation

The metrics I currently used is "accuracy"
