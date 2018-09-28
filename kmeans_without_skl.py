
# coding: utf-8

# In[16]:


import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[17]:


from sklearn.datasets.samples_generator import make_blobs


# In[18]:


x,y_true = make_blobs(n_samples = 300, centers= 4, cluster_std= 0.8, random_state= 101)


# In[19]:


plt.scatter(x[:,0], x[:,1], s= 50, );


# In[37]:


from sklearn.cluster import KMeans


# In[49]:


kmeans = KMeans(n_clusters = 4)


# In[50]:


kmeans.fit(x)


# In[51]:


y_means= kmeans.predict(x)
print(y_means)


# In[55]:


from sklearn.metrics import pairwise_distances_argmin


# In[59]:


def find_cluster(x, n_clusters,rseed=2):
    ## randomly choose cluster
    
    rng = np.random.RandomState(rseed)
    i = rng.permutation(x.shape[0])[:n_clusters]
    centers= x[i]
    while True:
        ## 2a. Assign labels on closest centers
        labels = pairwise_distances_argmin(x, centers)
        
        ##2b. find new centers from means of points
        new_centers =np.array([x[labels== i].mean(0)
                               for i in range(n_clusters)])
        
        ##2c. check for convergence
        if np.all(centers == new_centers):
            break
        centers = new_centers
        
    return centers, labels

centers, labels = find_cluster(x, 4)
plt.scatter(x[:,0], x[:,1], c= y_means, s= 50, cmap ="viridis")

plt.scatter (centers[:, 0], centers[:, 1], c="black", s= 200, alpha =0.5);


