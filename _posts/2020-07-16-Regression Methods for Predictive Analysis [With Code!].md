---
Title: "Regression Methods for Predictive Analysis"
Date: 2020-07-14
Tags: [machine learning, data science, regression, linear-regression, ridge-regression, lasso-regression, regularization]
header:
  image: "/images/regression.png"
excerpt: "Understanding the different regression models used for prediction"
---


# What are the different types of Predictive Regression Methods for Analysis?

When working we data, we try to understand the relationship between the various data points. Regression usually does not go unmentioned when deciding how to determine these relationships.

It helps us determine the relationship between the dependent variable(s)[output variable(s)] and independent variable(s)[input variable(s)].
The number of variables - input or output could be one or many. The most commonly used regression models have either one input and one output vairable, two or more input and one output variable. When there is one or more input variable and two or more output variable, it is considered to be a multivariate regression problem.

This article covers regression methods for univariate(single output) problems. Those considered are;

- Linear Regression
- Lasso Regression
- Ridge Regression
- ElasticNet Regression


## TO-DO before and after any Regression task

Make sure to have all these points checked before starting and after finishing any regression task or any machine learning related project.

#### Before 

- You have dealt with all outliers in your dataset. Outliers are datapoints in your dataset that are usually very different from the rest of the data. They might have gotten there due to an error during collection.

- Your dataset does not have any missing values, or if it does you fill them or delete the corresponding rows/colunms if they are irrelevant to your prediction.

#### After 

- The model does not overfit and underfit.


## Linear Regression

This is the most popular type of regression which has only linear variables. Used for simpler modelling problems that do not involve large datasets - mostly problems that have;

- one input and one output vairable
- two or more input and one output variable

<img src="{{ site.url }}{{ site.baseurl }}/images/lin_reg.png" alt="Linear regression">


The Boston Housing dataset is used here and is already available in the [sklearn python module](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.datasets). [Scikit Learn](https://scikit-learn.org/stable/) is a machine learning framework.


### Python code for Linear Regression 


```

# Train model 

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

model = LinearRegression()
model.fit(X, data.PRICE)


# Now predicting 

model.predict(X)[0:5]


# Checking for the error( in this case, Mean Squared Error)

MSE = mean_squared_error(data.PRICE, model.predict(X))
print MSE

```


## Lasso Regression

This model is used for data that has high collinearity among the feature variables and also used to avoid overfitting by not generating high coefficients for predictors. Collinearity is when there is a linear relationship between some features in the dataset. It helps reduce the collinearity by adding a small bias to the sum of the absolute values of coefficients which greatly reduces the variance. Lasso regression uses the L1 regularization for this by tending the sum of the errors towards zero using the bias(tuning parameter), *`alpha`*. The higher the value of *`alpha`*, the more the error tends to zero. Cross validation is one technique that helps in choosing the right value of *`alpha`*.


### Python code for Lasso Regression


For this method we would be using the [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) library to enable us use a range of different regularization parameters in order to find the optimal value of *`alpha`*

```
# Train model 

from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

model = Lasso()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]}

lasso_regressor = GridSearchCV(model, parameters, scoring = 'neg_mean_squared_error', cv = 5)

lasso_regressor.fit(X, data.PRICE)


# Checking for the error( in this case, Mean Squared Error)

MSE = mean_squared_error(data.PRICE, model.predict(X))
print MSE

```


## Ridge Regression

This regression model is very similar to Lasso Regression except that here, we add the small bias, *`alpha`* to the sum of the squares of coefficients. This is known as the L2 regularization.


### Python code for Ridge Regression

For this method we would be using the [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) library to enable us use a range of different regularization parameters in order to find the optimal value of *`alpha`* instead of setting the value of *`alpha`* randomly.

```
# Train model 

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

model = Ridge()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]}

ridge_regressor = GridSearchCV(model, parameters, scoring = 'neg_mean_squared_error', cv = 5)

ridge_regressor.fit(X, data.PRICE)


# Checking for the error( in this case, Mean Squared Error)

MSE = mean_squared_error(data.PRICE, model.predict(X))
print MSE

```

The **`neg_mean_squared_error`** is used as the scoring function in the code because it handles both the loss function and the scoring function. For the loss function, a smaller value is better, while for a scoring function (like F1 score/F2 score), a higher value is better. Therefore, they always return the negative, to avoid rewriting the function depending on each case.


#### Some noticable differeces between the L1 and L2 regularizations include;

- Lasso reduces the coefficients of less important features further to zero, this helps eliminate some very unimportant features. It is said to have a built-in feature selection mechanism.

- Ridge regressor makes the relevance of all the features even, including features that are not so relevant.

- For these 2 reasons above, L2 is more computationally efficient than L1 because they have an analytical solution that L1 doesn't.

- They both seem to have their pros and cons. How about taking the pros of both and forming an algorithm?


## ElasticNet Regression

This regresssion method combines the Lasso and Ridge reression - combination of both L1 and L2 regularization which makes it more preferrable for use.


### Python code for ElasticNet Regression

The new addition to the ElasticNet Regression code that makes it different from Lasso and Ridge is the *`l1_ratio`*. It is used to set how closer to Lasso or Ridge the ElasticNet Regression is. When the *`l1_ratio`* is set to 0 it is the same as ridge regression and when set to 1 it is lasso. Elastic net is somewhere between 0 and 1 when *`l1_ratio`* is set.


```
# Train model 

from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

model = ElasticNet()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]}
l1_ratio = [0.2, 0.4, 0.6, 0.8]

elasticnet_regressor = GridSearchCV(model, parameters, l1_ratio, scoring = 'neg_mean_squared_error', cv = 5)

elasticnet_regressor.fit(X, data.PRICE)


# Checking for the error( in this case, Mean Squared Error)

MSE = mean_squared_error(data.PRICE, model.predict(X))
print MSE

```


That's it! You can now implement your own different regression models. Check for the full code in this [repository]()

Don't hesitate to ask any questions you have in the comment section!😁
