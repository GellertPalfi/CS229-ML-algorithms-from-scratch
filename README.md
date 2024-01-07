![build status](https://github.com/GellertPalfi/CS229-ML-algorithms-from-scratch/actions/workflows/pyton-test.yml/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# CS229 Machine Learning Algorithms
Concise implementations of fundamental machine learning algorithms from scratch from [Stanford's CS229 course](https://www.youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU).  
The purpose of this project was to deepen my understanding of the most commonly used ML algorithms.

# Table of contents:
 - [Visualizations](#visualizations)
 - [Disclaimer](#disclaimer)
 - [Tests](#tests)
 - [Prerequisites](#prerequisite)
 - [Linear regression](#lin-reg)
 - [Logistic regression](#log-reg)
 - [Naive Bayes](#naive-bayes)
 - [Support Vector Machine](#svm)
 - [Simple Neural Networks](#simple-NN)
 - [Backpropagation](#back-prop)
 - [Batch Gradient Descent](#grad-desc)
 - [Resources and useful links](#useful-links)

<a id="visualizations"></a>
# Visualizations:
Because looking at code is not the most interesting thing, here are some visualizations (Code included in the repo) that I made with my implementations of the algorithms.

### Linear regression line fit:
![lin_reg_fit](https://github.com/GellertPalfi/CS229-ML-algorithms-from-scratch/assets/69762257/9779e50f-5ed8-4f6c-80a7-27a4fb98fcb2)

### Gradient descent searching for optimal parameters on the error surface in the weight space:
![grad_descent_progression](https://github.com/GellertPalfi/CS229-ML-algorithms-from-scratch/assets/69762257/8003ae93-2b56-4ae1-9ed9-5e18c38fa297)






<a id="disclaimer"></a>
# Disclaimer
These algorithms are very simple and primitive implementations of those found in popular ml packages such as [sci-kit learn](https://scikit-learn.org/stable/), which have been refined and optimized by experts for years, so any of the implementations here should only be used for learning purposes.  

<a id="tests"></a>
# Tests
All algorithms and helper functions are fully tested with a coverage of: [coverage badge]

<a id="prerequisite"></a>
# Prerequisites
To run this project you will need to have python installed on your machine and (preferably) a virtual enviroment.

<a id="install"></a>
# Install
Installation steps:
1. activate your virtual enviroment
2. install necessary libraries:  
```pip install -r requirements.txt```
3. run any of the scripts 

<a id="lin-reg"></a>
# Linear regression
[Linear regression](https://en.wikipedia.org/wiki/Linear_regression) is a statistical model which tries to model the relationship beetween the [dependent and independent variables](https://en.wikipedia.org/wiki/Dependent_and_independent_variables).  
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
[Logistic regression](https://en.wikipedia.org/wiki/Logistic_regression) is a statistical model which works by applying the [logistic function](https://en.wikipedia.org/wiki/Logistic_function) to the linear relationship combined from the input features, weights and biases. Mostly used for classification with a given treshold (usually 0.5), where values returned by the logistic function greater than or equal to the treshold are classified as 1 , below the treshold classified as 0.  

Logistic regression is most commonly trained by minimizing the negative of the log likelihood:  
![image](https://github.com/GellertPalfi/CS229-ML-algorithms-from-scratch/assets/69762257/18b1fb19-b291-4bca-b81e-236f616bbb15)

Training and then comparing my results to sklearn yields similar results:  
![image](https://github.com/GellertPalfi/CS229-ML-algorithms-from-scratch/assets/69762257/5b701bc9-7da8-402e-8855-75aeb730a469)  
As you can see the weights are of by a little bit. This is because my algorithm haven't converged yet. The log likelihood is a concave function, meaning any local optimum is also the global one.
This means my algorithm would eventually reach the same weights as sklearn given enough iterations.



<a id="naive-bayes"></a>
# Naive Bayes
[Naive Bayes classifier](https://en.wikipedia.org/wiki/Naive_Bayes_classifier) is a probabilistic classifier based on [Bayes' theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem) and the **naive** assumption
that the features are [independent](https://en.wikipedia.org/wiki/Independence_(probability_theory)). Although the features are rarely independent naive bayes is nevertheless used in real situations because of its speed and the fact that it gives surprisingly good results. Naive bayes differs from the usual ml algorithm because for the training a closed-form solution can be used rather than an expensive iterative algorithm such as [gradient descent](#grad-desc). I implemented a Gaussian Naive bayes classifier which assumes that the features are independent, continous variables and they follow a normal distribution.

For this you only need to calculate the mean, variance and prior probability of each class(here i used [polars](https://pola.rs): 
![image](https://github.com/GellertPalfi/CS229-ML-algorithms-from-scratch/assets/69762257/989bf45e-9017-47cb-8b76-7d386d05dd44)  

After this any new prediction can be made by pluggint these variables into the [Probability density function](https://en.wikipedia.org/wiki/Probability_density_function) and returing the label with the higest probablity:  
![image](https://github.com/GellertPalfi/CS229-ML-algorithms-from-scratch/assets/69762257/500addef-663f-4957-a2fc-aa3ccaa58875)  





<a id="svm"></a>
# Support Vector Machine

<a id="simple-NN"></a>
# Simple Neural Networks

<a id="back-prop"></a>
# Backpropagation
Writing here about [backpropagation](https://en.wikipedia.org/wiki/Backpropagation) without going into the depths and writing 5+ pages is kinda hard, so here is a *really* is a short explanation: You calculate the error at the end of the NN and using the chain rule to calculate errors in previous layers. However this creates more problems such as: [vanishing gradient](https://en.wikipedia.org/wiki/Vanishing_gradient_problem) or the opposite [exploding gradient](https://deepai.org/machine-learning-glossary-and-terms/exploding-gradient-problem)

<a id="grad-desc"></a>
# Batch Gradient Descent
For us to understand gradient descent, first we need to know what the gradient is. According to wikipedia:  
![gradient](https://github.com/GellertPalfi/CS229-ML-algorithms-from-scratch/assets/69762257/89c4eab7-8327-4323-bbe5-b724df9a4763)


<a id="useful-links"></a>
# Resources used and useful links
- [CS229 lecture notes](https://cs229.stanford.edu/main_notes.pdf)
- [CS229 YT playlist](https://www.youtube.com/watch?v=jGwO_UgTS7I&list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU)
- [Logistic Regression from Scratch in Python](https://beckernick.github.io/logistic-regression-from-scratch/)
- [Building a Neural Network from Scratch in Python and in TensorFlow](https://beckernick.github.io/neural-network-scratch/)
- [svm-from-scratch](https://www.kaggle.com/code/prabhat12/svm-from-scratch)
- [The Hundred-Page Machine Learning Book by Andriy Burkov](https://themlbook.com/)
