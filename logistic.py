
# importing the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# get_ipython().run_line_magic('matplotlib', 'inline')


# reading teh dataset using pandas library
train = pd.read_csv("titanic_train.csv")
train.head()


# checking for any null values in the dataset
train.isnull()

# discarding the cabin column from the dataset
train.drop('Cabin', axis=1, inplace=True)
train.head()


# replacing the null values of age column by the average of age
based on the class of passengers


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    
    else:
        return Age




# applying the calculated ages to the dataset
train["Age"]= train[["Age", "Pclass"]].apply(impute_age, axis= 1)
train.isnull()


# replacing the categoricals values by binary values of 0 or 1
sex = pd.get_dummies(train["Sex"], drop_first= True)
embark = pd.get_dummies(train["Embarked"], drop_first= True)


# Adding the binary columns to the dataset
pd.get_dummies(train["Embarked"], drop_first= True)
train.drop(["Sex", "Embarked", "Name", "PassengerId", "Ticket"], axis=1, inplace= True)
train= pd.concat([train, sex, embark],axis=1)
train.head()


# importing training and testing split library form scikit-learn
from sklearn.model_selection import train_test_split


# spliting the data into traing and testing sets
X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1),train['Survived'], test_size=0.30, random_state=101)


# Importing logisticregression function
from sklearn.linear_model import LogisticRegression
logmodel= LogisticRegression()
logmodel.fit(X_train, y_train)


# predicting the value of testing data using logistic regression
predictions = logmodel.predict(X_test)


# importing classification report library for checking the accuracy of predicted model 
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))

