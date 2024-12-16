# General Class Balancer

This program finds a subset of your dataset with balanced confounding factors (also known as "data matching"), though it can work with any combination of categorical and continuous variables.

Given a labeled dataset with any number of classes and a number of confounding factors for each datapoint, this matches data in each class to one another, such that the distributions of each confounding factor are the same in each class. This may be used to sample a training set on which a given deep learning model will not take confounding factors into account during its classification.

A version of this method was originally introduced in https://arxiv.org/abs/2002.07874 to ensure that deep learning classifications of sex based on brain activity did not take into account head motion or intracranial volume, both of which are statistically different between sexes and which affect measurements of brain activity, but which we did not want the machine learning model to consider. A more detailed explanation of the method may be found in that paper. Below is a pictural description of the algorithm (assuming plim = 0.10)

![alt text](description.png "A description of the general class balancer algorithm")

Everything in the presented code uses numpy arrays. The code, as well as a script that simulates data from random variables, is given. Simply run


```
python random_example_test.py
```
