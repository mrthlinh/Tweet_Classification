# Tweet Sentiment Classification

----------

In this part, I work with a set of Tweets about US airlines and examine their sentiment polarity.
More details about thedataset can be found on the website [Kaggle](https://www.kaggle.com/crowdflower/twitter-airline-sentiment). My aim is to learn to classify Tweets
as either \positive", \neutral", or \negative" by using two classifiers and pipelines for pre-processing
and model building.

All this has to be done in a Scala class, which has to be part of a Scala SBT or Maven project. Make
sure you have all your dependencies and the class can be run on AWS. The class will have 2 parameters
one that represents the path of the input file and the second one that represents the output path
where the output will be stored.

Below are the steps of the project:

1. **Loading**: First step is to define an input argument that defines the path from which to load the dataset. After that, you will need to remove rows where the text field is null.