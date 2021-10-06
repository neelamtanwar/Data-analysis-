#!/usr/bin/env python
# coding: utf-8

# Linear Regression is one of the most fundamental algorithms in the Machine Learning world. It is the door to the magical world ahead.
# But before proceeding with the algorithm, let’s first discuss the lifecycle of any machine learning model. This diagram explains the creation of a Machine Learning model from scratch and then taking the same model further with hyperparameter tuning to increase its accuracy, deciding the deployment strategies for that model and once deployed setting up the logging and monitoring frameworks to generate reports and dashboards based on the client requirements. 
# A typical lifecycle diagram for a machine learning model looks like:
# 
# <img src="MLApplicationFlow_bold.PNG" width= "300">
# 
# Now, let's take our discussion of Linear Regression further
# 
# ## What is Regression Analysis?
# 
# Regression in statistics is the process of predicting a Label(or Dependent Variable) based on the features(Independent Variables) at hand. Regression is used for time series modelling and finding the causal effect relationship between the variables and forecasting. For example, the relationship between the stock prices of the company and various factors like customer reputation and company annual performance etc. can be studied using regression.
# 
# 
# Regression analysis is an important tool for analysing and modelling data. Here, we fit a curve/line to the data points, in such a manner that the differences between the distance of the actual data points from the plotted curve/line is minimum. The topic will be explained in detail in the coming sections.
# 
# 
# ## The use of Regression
# 
# Regression analyses the relationship between two or more features. Let’s take an example:
# 
# Let’s suppose we want to make an application which predicts the chances of admission a student to a foreign university. In that case, the 
# 
# The benefits of using Regression analysis are as follows:
# 
#    * It shows the significant relationships between the Lable (dependent variable) and the features(independent variable).
#    * It shows the extent of the impact of multiple independent variables on the dependent variable.
#    *  It can also measure these effects even if the variables are on a different scale.
# 
# These features enable the data scientists to find the best set of independent variables for predictions.
# 
# 
# ## Linear Regression
# 
# Linear Regression is one of the most fundamental and widely known Machine Learning Algorithms which people start with. Building blocks of a Linear Regression Model are:
# * Discreet/continuous independent variables
# * A best-fit regression line
# * Continuous dependent variable.
# i.e., A Linear Regression model predicts the dependent variable using a regression line based on the independent variables.
# The equation of the Linear Regression is:
# 
#                                                 Y=a+b*X + e 
# 
#  Where,
#  a is the intercept, 
# b is the slope of the line, 
# and e is the error term. 
# The equation above is used to predict the value of the target variable based on the given predictor variable(s).
# 
# 
# ### The Problem statement:
# 
# This data is about the amount spent on advertising through different channels like TV, Radio and Newspaper. The goal is to predict how the expense on each channel affects the sales and is there a way to optimise that sale?
# 
# 
# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv("D:\project/Advertising.csv")


# In[3]:


data.head()


# features
# Tv: advertising dollar spent on TV commercials
# radio : dollars spent on Radio commercials 
# newspaper:dollars spent on radio commercials
# sales:sale of particular product in market
# 

# In[4]:


data.shape


# In[5]:


data.info()


# In[6]:


data.isna().sum() #no null value


# # VISUALISATION

# In[7]:


#finding the relationship
fig, axs = plt.subplots(1, 3, sharey=True)
data.plot(kind='scatter',x='TV',y='sales',ax=axs[0],figsize=(16,8))
data.plot(kind='scatter',x='radio',y='sales',ax=axs[1],figsize=(16,8))
data.plot(kind='scatter',x='newspaper',y='sales',ax=axs[2],figsize=(16,8))


# here we could see that variablity of sales data is more when advertising through nespaper and least in case of TV.
# Here also we could attend same amount of sales via radio with small expenditure as compare to tv.
# 

# now a generic question is? how compnyshould optimise .
# Here we could find that relationship between sales and TV and sales and radio is nearly linear so we can use linear regression.

# In[25]:


#predicting for TV only
feature_cols = ['TV']
x=data[feature_cols]
y=data.sales

#importing linear regression
from sklearn.linear_model import LinearRegression 
lm = LinearRegression()
lm.fit(x,y)
print(lm.intercept_)
print(lm.coef_)


# A unit increase in tv ad will increase 0.0475 units increase in sales

# In[24]:


0.04753664*50+7.032593549127693 #predictiing the value


# In[21]:


X_new = pd.DataFrame({'TV':[50]})
X_new.head()


# In[22]:


lm.predict(X_new)


# In[28]:


#plotting the fitted line
#Creatting a new data frame that include minimum and maximum sales value for TV
X_new = pd.DataFrame({'TV': [data.TV.min(), data.TV.max()]})
X_new.head()


# In[33]:


#predicting the values
predicted=lm.predict(X_new)
predicted


# In[36]:


#plotting the data 
data.plot(kind='scatter',x='TV',y='sales')
#plottin the line
plt.plot(X_new,predicted,c='Red',linewidth=2)


# In[37]:


import statsmodels.formula.api as smf
lm = smf.ols(formula='sales~ TV',data=data).fit()
lm.conf_int()
lm.summary()


# Is it a "good" R-squared value? Now, that’s hard to say. In reality, the domain to which the data belongs to plays a significant role in deciding the threshold for the R-squared value. Therefore, it's a tool for comparing different models.

# In[38]:


#multiple linear regression model


# In[8]:


#creating x and y 
feature_cols = ['TV','radio','newspaper']
x=data[feature_cols]
y=data.sales

#importing linear regression
from sklearn.linear_model import LinearRegression 
lm = LinearRegression()
lm.fit(x,y)
print(lm.intercept_)
print(lm.coef_)


# here newspaper coeffecient is negative , it implies it does not provide positive sales when advertising through newspaper.
# 

# In[15]:


#model summary output
import statsmodels.formula.api as smf
lm = smf.ols(formula='sales~ TV + radio + newspaper',data=data).fit()
lm.conf_int()
lm.summary()


# null hypothesis = There is no relationship between features and sales
# here we can find that our p values has positve value, therefore we could reject our null hypothesis.
# but we fall to reject this null hypothesis for newspaper.
# The expense on both radio and tv are positively asssociated with sales , while expense on newspaper are slightly associated negatively with sales.
# This model has a higher value of R-squared (0.897) than the previous model, which means that this model explains more variance and provides a better fit to the data than a model that only includes the TV.
# 

# In[ ]:




