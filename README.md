# titanic_ml
repository containing notebooks and code for creating an xgboost model using titanic data. 

Note that all notebooks and files assume that the titanic data can be found in titanic_ml/data/titanic/.

This includes the following sections:

## notebooks
directory containing example steps of a model building process:

### 1_explore_data
notebook with some exploratory analysis of the dataset, some examples of data cleaning and feature engineering.

### 2. Feature Selection
notebook with some examples of things to consider when selecting features to use in a modelling process

### 3. make a model
notebook with a simple example of training a model, and generating some statistics on the model performance (AUC, shap values, PSI analysis)

## api
This directory contains everything needed to set up the fast-api, and query the model as using HTTP GET RESTful queries. 

### make_model.py
File that takes the data and creates the model pipelines that the api will use. this will create a 'model' directory, with the exported model inside it. You must run this file to set up for the docker image.

### main.py
File that defines the api parameters, object etc.

## src
This directory contains a functions.py file that has various helper functions that are used in the notebooks and python files.

## requirements.txt
use this to install all necessary packages for the code to run.

## Dockerfile
This contains the instructions needed to set up the docker image for this project. see below for a brief setup instruction:

1) navigate to the titanic_ml directory, and run docker build --tag titanic-docker .
2) start the container: docker run -d --name titanic-docker -p 80:80 titanic-docker
3) there will now be a service running on http://127.0.0.1:80
4) navigate to http://127.0.0.1/docs - this will show the documentation for the api. 
5) you can try it out by clicking the post /score dropdown, and then clicking 'try it now'
