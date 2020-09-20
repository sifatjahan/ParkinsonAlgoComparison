#!/usr/bin/env python
# coding: utf-8

# In[122]:


import numpy as np
import pandas as pd
import os, sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[123]:


#Read the data
df=pd.read_csv('D:\\dataset\\parkinsons.data')
df.head()


# In[124]:


#Get the features and labels
features=df.loc[:,df.columns!='status'].values[:,1:]
labels=df.loc[:,'status'].values


# In[125]:


#Scale the features to between -1 and 1
scaler=MinMaxScaler((-1,1))
x=scaler.fit_transform(features)
y=labels


# In[149]:


#Split the dataset
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.50, random_state=0)


# In[150]:


print(Counter(y_train))
print(Counter(y_test))


# In[151]:


#Train the model

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(x_train,y_train)


# In[152]:


DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                       max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort=False,
                       random_state=None, splitter='best')


# In[153]:


#Calculate the accuracy
y_pred=model.predict(x_test)
print(accuracy_score(y_test, y_pred)*100)


# In[154]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))


# In[ ]:





# In[ ]:




