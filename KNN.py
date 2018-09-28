
# Importing the necessary libaries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#get_ipython().run_line_magic('matplotlib', 'inline')


# reading the dataset

df = pd.read_csv("Classified Data", index_col=0)
df.head()

# scaling the variables of dataset between vlues of -1 to 1
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS',axis=1))
scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))


# using python's pandas library to form dataframe of set excluding target column 
df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
df_feat.head()

# spliting the dataset into training and testing set
from sklearn.model_selection import train_test_split
X = df_feat
y = df["TARGET CLASS"]
X_train, X_test, y_train, y_test = train_test_split(scaled_features,df['TARGET CLASS'],
                                                    test_size=0.33, random_state = 101)


# Importing the K-nearest neighbouring library from sciket-learn.
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors= 1)


# training the model for find k-nearest neighbour
knn.fit(X_train, y_train)


#  predicting the neighbours of test dataset
pred = knn.predict(X_test)


# importing the confusion matrix for comparing the preicted neighbour to original neighbour of the test dataset
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))

