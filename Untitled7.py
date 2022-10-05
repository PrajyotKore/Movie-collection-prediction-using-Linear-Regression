#!/usr/bin/env python
# coding: utf-8

# In[5]:


import sys
get_ipython().system('{sys.executable} -m pip install category_encoders')


# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn import preprocessing 
from category_encoders import *
from sklearn.preprocessing import LabelEncoder
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model, metrics
from sklearn.metrics import  confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, plot_confusion_matrix


# In[16]:


import os 


# In[19]:


os.getcwd()


# In[23]:


os.chdir("C:/Users/Prajyot_kore/OneDrive/Desktop/Python/Project/Movies regression")


# In[28]:


dataframe=pd.read_csv('C:/Users/Prajyot_kore/OneDrive/Desktop/Python/Project/Movies regression/Movie_regression.xls')


# # Exploratory Data Analysis

# In[29]:


dataframe.head()


# In[31]:


dataframe.tail()


# In[34]:


dataframe.shape


# In[36]:


dataframe.dtypes


# ### Columns of the Dataset

# In[38]:


dataframe.columns


# ### Summary Statistics for dataset

# In[40]:


dataframe.info()


# In[42]:


dataframe.describe()


# ### Checking for Dublicates and Null values

# In[44]:


dataframe.duplicated().sum()


# In[46]:


dataframe.isnull().sum()


# ### Correlation Matrix for Given Data

# In[48]:


dataframe.corr()


# ### Skewness for Given Dataset Columns

# In[50]:


dataframe.skew()


# In[52]:


dataframe['3D_available'].value_counts()


# ### CountPlot for Column "3D_available"

# In[57]:


sns.countplot(x ='3D_available', data=dataframe)
plt.show()


# ### Histogram plot for Average Collection and Counts

# In[59]:


fig = px.histogram(dataframe, 'Collection',             
                   color="3D_available",
                   title="<b>Average Collection</b>")

fig.add_vline(x=dataframe['Collection'].mean(), line_width=2, line_dash="dash", line_color="black")

fig.show()

#For collections morethan 50k majority of the collection came for 3D movies


# ### Countplot for Column Genre

# In[61]:


dataframe['Genre'].value_counts()


# In[63]:


sns.countplot(x = 'Genre',data=dataframe)
plt.show()


# ### Histogram for column "Collection"

# In[65]:


fig = px.histogram(dataframe, 'Collection',             
                   color="Genre",
                   title="<b>Average Collection</b>")

fig.add_vline(x=dataf['Collection'].mean(), line_width=2, line_dash="dash", line_color="black")

fig.show()


# ### Histogram for all columns except 3D_available and Genre

# In[67]:


plt.figure(figsize=(6,8))
x = dataframe.drop(['3D_available','Genre'],axis = 1)
for i in x.columns:
    sns.histplot(x[i],kde = True)
    plt.show()


# ### Scatterplot for scatterplot of Collection vs other columns of data except '3D_available','Genre'

# In[119]:


plt.figure(figsize=(6,8))
x = dataframe.drop(['3D_available','Genre'],axis = 1)
for i in x.columns:
    sns.scatterplot(x = 'Collection',y = i,data = dataframe,color = 'Red')
    plt.show()


# ### Heatmap of Correlation for Columns

# In[121]:


plt.figure(figsize=(16,9))
x = dataframe.drop(['3D_available','Genre'],axis = 1)
ax = sns.heatmap(x.corr(),annot = True,cmap = 'viridis')
plt.show()


# ### Plot a pairwise relationships in a dataset

# In[76]:


sns.pairplot(dataframe)


# ### Plotting boxplot for all Columns 

# In[79]:


x = dataframe.drop(['3D_available','Genre'],axis = 1)
for i in x.columns:
    sns.boxplot(x = i, data = x,color = 'yellowgreen')
    plt.xlabel(i)
    plt.show()


# ### Violin plots for all columns expects '3D_available','Genre'

# In[81]:


x = dataframe.drop(['3D_available','Genre'],axis = 1)
for i in x.columns:
    sns.violinplot(x = i, data = x,color = 'yellowgreen')
    plt.xlabel(i)
    plt.show()


# In[86]:


def count_outliers(data,col):
        q1 = data[col].quantile(0.25,interpolation='nearest')
        q2 = data[col].quantile(0.5,interpolation='nearest')
        q3 = data[col].quantile(0.75,interpolation='nearest')
        q4 = data[col].quantile(1,interpolation='nearest')
        IQR = q3 -q1
        global LLP
        global ULP
        LLP = q1 - 1.5*IQR
        ULP = q3 + 1.5*IQR
        if data[col].min() > LLP and data[col].max() < ULP:
            print("No outliers in",i)
        else:
            print("There are outliers in",i)
            x = data[data[col]<LLP][col].size
            y = data[data[col]>ULP][col].size
            a.append(i)
            print('Count of outliers are:',x+y)
global a
a = []
for i in x.columns:
    count_outliers(x,i)


# # Data Preprocessing
# 

# Treating with Null values

# In[90]:


dataframe.isnull().sum()


# ### Since there are outliers in time_taken column we should replace null with median

# In[93]:


dataframe['Time_taken'].fillna(dataframe['Time_taken'].median(),inplace=True)


# In[96]:


dataframe.isnull().sum()


# # Encoding

# In[123]:


dummi=pd.get_dummies(data=dataframe,columns=['Genre','3D_available'],drop_first=True)
label_2


# # Feature Selection

# In[124]:


X = dummi.drop(['Collection'],axis = 1)
Y = dummi['Collection']
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.3,random_state=44)


# # Prediction Using Linear Regression

# In[125]:


regression = linear_model.LinearRegression()
regression.fit(X_train, Y_train)


# In[126]:


#Regression Coeeficient
regression.coef_


# In[127]:


pred = regression.predict(X_test)
pred


# In[110]:


plt.scatter(Y_test,pred)
plt.xlabel('Y Test (True Values)')
plt.ylabel('Predicted values')
plt.show()


# In[111]:


print('MAE',metrics.mean_absolute_error(Y_test,pred))
print('MSE',metrics.mean_squared_error(Y_test,pred))
print('RMSE',np.sqrt(metrics.mean_squared_error(Y_test,pred)))


# In[113]:


#r2 score
metrics.explained_variance_score(Y_test,pred)


# In[115]:


# Curve is distributed normally so model is ok
sns.displot(Y_test-pred,bins = 50,kde = True)


# ### This says that for 1 unit increase in collection marketing expense will decrease by -12.442049 units

# In[116]:


cdf = pd.DataFrame(reg.coef_,X.columns,columns = ['coef'])
cdf


# In[ ]:




