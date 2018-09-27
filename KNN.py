
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


df = pd.read_csv("Classified Data", index_col=0)


# In[9]:


df.head()


# In[10]:


from sklearn.preprocessing import StandardScaler


# In[11]:


scaler = StandardScaler()


# In[12]:


scaler.fit(df.drop('TARGET CLASS',axis=1))


# In[13]:


scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))


# In[14]:


df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
df_feat.head()


# In[15]:


from sklearn.model_selection import train_test_split


# In[17]:


X = df_feat
y = df["TARGET CLASS"]

X_train, X_test, y_train, y_test = train_test_split(scaled_features,df['TARGET CLASS'],
                                                    test_size=0.33, random_state = 101)


# In[18]:


from sklearn.neighbors import KNeighborsClassifier


# In[20]:


knn = KNeighborsClassifier(n_neighbors= 1)

knn.fit(X_train, y_train)


# In[21]:


pred = knn.predict(X_test)


# In[22]:


from sklearn.metrics import classification_report,confusion_matrix


# In[23]:


print(confusion_matrix(y_test,pred))


# In[24]:


print(classification_report(y_test,pred))

