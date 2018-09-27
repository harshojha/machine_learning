
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


from sklearn.datasets.samples_generator import make_blobs


# In[4]:


x,y_true = make_blobs(n_samples = 300, centers= 4, cluster_std= 0.8, random_state= 101)


# In[6]:


plt.scatter(x[:,0], x[:,1], s= 50, );


# In[8]:


from sklearn.cluster import KMeans


# In[9]:


kmeans = KMeans(n_clusters = 4)


# In[10]:


kmeans.fit(x)


# In[11]:


y_means= kmeans.predict(x)
print(y_means)

