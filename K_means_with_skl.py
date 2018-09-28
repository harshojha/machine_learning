
# importing the necessary libraries


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# importing sciket-learn's in-bulit blobs dataset
from sklearn.datasets.samples_generator import make_blobs

# formation of cluster
x,y_true = make_blobs(n_samples = 300, centers= 4, cluster_std= 0.8, random_state= 101)


# ploting the scatter plot of cluster

plt.scatter(x[:,0], x[:,1], s= 50, );


#importing KMeans library from scikit-learn

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 4)


# Impleting the Kmeans on training set (x)
kmeans.fit(x)

# predicting the cluster to which y belongs
y_means= kmeans.predict(x)
print(y_means)

