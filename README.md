# CS229 Machine Learning Algorithms
Concise implementations of fundamental machine learning algorithms from scratch from [Stanford's CS229 course](https://www.youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU).  
The purpose of this project was to deepen my understanding of the most commonly used ML algorithms.

# Table of content:
 - [Visualizations](#visualizations)
 - [Disclaimer](#disclaimer)
 - [Prerequisites](#prerequisite)
 - [Linear regression](#lin-reg)
 - [Logistic regression](#log-reg)
 - [Naive Bayes](#naive-bayes)
 - [Support Vector Machine](#svm)
 - [Simple Neural Networks](#simple-NN)
 - [Backpropagation](#back-prop)
 - [Batch Gradient Descent](#grad-desc)
 - [Useful links](#useful-links)

<a id="visualizations"></a>
# Visualizations:
Because looking at code is not the most interesting thing,  
here are some visualizations (Code included in the repo) that I made with my implementations of the algorithms.

### Linear regression line fit:
![lin_reg_fit](https://github.com/GellertPalfi/CS229/assets/69762257/a4daed4c-1753-45c6-9de3-f09c07763de1)

### Gradient descent searching for optimal parameters on the error surface in the weight space:
![grad_descent_progression](https://github.com/GellertPalfi/CS229/assets/69762257/e59efabd-494e-4515-b9bb-4dfd7c9b42e7)

<a id="disclaimer"></a>
# Disclaimer
These algorithms are very simple and primitive implementations of those found in popular ml packages such as sci-kit learn, which have been refined and optimized by experts for years, so any of the implementations here should only be used for learning purposes.  
All algorithms and helper functions are fully tested and the results are compared to popular ml package implementations of the same algorithm.

<a id="prerequisite"></a>
# Prerequisites
To run this project you will need to have python installed on your machine and a virtual enviroment.

<a id="install"></a>
# Install
Installation steps:
1. activate your virtual enviroment
2. install necessary libraries:  
```pip install -r requirements.txt```
3. run any of the scripts with the data already provided or with your own

<a id="lin-reg"></a>
# Linear regression
[Linear regression](https://en.wikipedia.org/wiki/Linear_regression) works by trying to model the relationship beetween the [dependent and independent variables](https://en.wikipedia.org/wiki/Dependent_and_independent_variables).  
It is mostly used for predicting continous values and rarely for classification as it is really sensitive to outliers.  
Altough a closed-form solution exits to linear regression, which would give you the optimal parameter values directly, I still used gradient descent to gain deeper knowledge of the algorithm.  
  
The error metric that we are trying to minimalize is the [root](https://en.wikipedia.org/wiki/Root-mean-square_deviation) (RMSE) or the normal [mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error) (MSE):  
![image](https://github.com/GellertPalfi/CS229/assets/69762257/09f93ae5-abdd-4d09-bec6-468b3d835412)  

Running the algorithm for 10k iterations with a learning rate of `0.003` the parameters almost match the ones calculated by sklearn:  
![image](https://github.com/GellertPalfi/CS229/assets/69762257/c96d8e6e-c810-42b0-aab0-e816910da6a9)  
Since MSE is a convex function which means that the local optimum is also the global one, my algorithm would 
eventually converge given enough time.



<a id="log-reg"></a>
# Logistic regression

<a id="naive-bayes"></a>
# Naive Bayes

<a id="svm"></a>
# Support Vector Machine

<a id="simple-NN"></a>
# Simple Neural Networks

<a id="back-prop"></a>
# Backpropagation

<a id="grad-desc"></a>
# Batch Gradient Descent

<a id="useful-links"></a>
# Useful links
