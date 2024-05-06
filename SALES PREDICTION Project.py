#!/usr/bin/env python
# coding: utf-8

# Author :- Ashutosh Kumar
# Batch :- April
# Domain :- Data Science
# Aim :- To Build a model that Sales Prediction Project.

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# In[3]:


df = pd.DataFrame(pd.read_csv("advertising.csv"))
df.head(20)


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


# Checking Null values
df.isnull().sum()*100/df.shape[0]
# There are no NULL values in the dataset, hence it is clean.


# In[9]:


# Outlier Analysis
fig, axs = plt.subplots(3, figsize = (5,5))
plt1 = sns.boxplot(df['TV'], ax = axs[0])
plt2 = sns.boxplot(df['Newspaper'], ax = axs[1])
plt3 = sns.boxplot(df['Radio'], ax = axs[2])
plt.tight_layout()


# In[10]:


# There are no considerable outliers present in the data


# In[11]:


sns.boxplot(df['Sales'])
plt.show()


# In[12]:


# Let's see how Sales are related with other variables using scatter plot.
sns.pairplot(df, x_vars=['TV', 'Newspaper', 'Radio'], y_vars='Sales', height=4, aspect=1, kind='scatter')
plt.show()


# In[13]:


# Let's see the correlation between different variables.
sns.heatmap(df.corr(), cmap="YlGnBu", annot = True)
plt.show()


# In[15]:


X = df['TV']
y = df['Sales']


# In[16]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, test_size = 0.3, random_state = 100)


# In[17]:


X_train.head()


# In[18]:


y_train.head()


# In[19]:


import statsmodels.api as sm


# In[20]:


# Add a constant to get an intercept
X_train_sm = sm.add_constant(X_train)

# Fit the resgression line using 'OLS'
lr = sm.OLS(y_train, X_train_sm).fit()


# In[21]:


lr.params


# In[22]:


print(lr.summary())


# In[23]:


plt.scatter(X_train, y_train)
plt.plot(X_train, 6.948 + 0.054*X_train, 'r')
plt.show()


# In[24]:


y_train_pred = lr.predict(X_train_sm)
res = (y_train - y_train_pred)


# In[25]:


fig = plt.figure()
sns.distplot(res, bins = 15)
fig.suptitle('Error Terms', fontsize = 15)                  
plt.xlabel('y_train - y_train_pred', fontsize = 15)         
plt.show()


# In[26]:


plt.scatter(X_train,res)
plt.show()


# In[27]:


# Add a constant to X_test
X_test_sm = sm.add_constant(X_test)

# Predict the y values corresponding to X_test_sm
y_pred = lr.predict(X_test_sm)


# In[28]:


y_pred.head()


# In[29]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# In[30]:


#Returns the mean squared error; we'll take a square root
np.sqrt(mean_squared_error(y_test, y_pred))


# In[31]:


r_squared = r2_score(y_test, y_pred)
r_squared


# In[32]:


plt.scatter(X_test, y_test)
plt.plot(X_test, 6.948 + 0.054 * X_test, 'r')
plt.show()


# In[ ]:




