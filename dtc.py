
# coding: utf-8

# In[10]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[11]:


df = pd.read_csv("kyphosis.csv")


# In[9]:


df.head()


# In[7]:


df.info()


# In[12]:


from sklearn.model_selection import train_test_split


# In[13]:


X= df.drop("Kyphosis", axis =1)

y = df['Kyphosis']


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state = 101)


# In[15]:


from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


# In[16]:


dtc = DecisionTreeClassifier(criterion = "entropy" , random_state = 101, max_depth= 4, min_samples_leaf= 5)
dtc.fit(X_train, y_train)


# In[19]:


predict = dtc.predict(X_test)


# In[25]:


from sklearn.metrics import accuracy_score


# In[28]:


print("Accuracy is ", accuracy_score(y_test, predict)*100)

