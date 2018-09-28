
# importing the necessary libraries


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
# get_ipython().run_line_magic('matplotlib', 'inline')


# importing the in-built make_blobs dataset sample of sci-kit learn library
from sklearn.datasets.samples_generator import make_blobs
x, y= make_blobs(n_samples = 40, centers =2, random_state= 101)


# seting the kernal as linaer for support vector machine


clf = svm.SVC(kernel= "linear", C=1)
clf.fit(x,y)


# ploting the svm on x-y plane
plt.scatter(x[:,0], x[:, 1], c=y, s=30, cmap = plt.cm.Paired)
plt.show()


# predicting the group to which the input from user belongs
new_data = [[-5,-3],[4,6]]
print(clf.predict(new_data))


