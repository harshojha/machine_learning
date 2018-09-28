
# Importing the necessary libaries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#reading the in-built dataset of pandas
df = pd.read_csv("kyphosis.csv")

# head of dataset
df.head()


# information regarding the dataset 
df.info()

# importing sciket-learn library for training the model
from sklearn.model_selection import train_test_split


# spliting the data in training and testing sets
X= df.drop("Kyphosis", axis =1)

y = df['Kyphosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state = 101)


# import decision tree classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


# decision tree classifier for training the data with entropy as feature
dtc = DecisionTreeClassifier(criterion = "entropy" , random_state = 101, max_depth= 4, min_samples_leaf= 5)
dtc.fit(X_train, y_train)

# using decision tree to predict unseen dataset
predict = dtc.predict(X_test)


# Measuring the accuracy of decision tree classifier
from sklearn.metrics import accuracy_score


# printing the accuracy in percentage 
print("Accuracy is ", accuracy_score(y_test, predict)*100)

