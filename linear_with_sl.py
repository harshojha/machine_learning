
# Importing the necessary libraries 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# importing the dataset using pandas library
df = pd.read_csv("Ecommerce Customers")
df.describe()


# intializing the independant and depandent variables
x= df['Length of Membership'].values
n= len(x)
x= x.reshape((n,1))

# the "lenght of membership" is independant and "yearly amount spend" is dependant variable
 
y = df['Yearly Amount Spent'].values


# splittinf the dataset in training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.4, random_state=101)


# importing the linear regression library from sciket-learn


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,Y_train)

# printing the value of Y-intercept
print(lm.intercept_)


# predicting the value of y for testing dataset
predict = lm.predict(X_test)


# ploting teh graph of actual and predicted values of dapendant variable 
plt.plot(predict)
plt.scatter(Y_test, predict)


# calculating the R-square value of model to explain the varience in depandent variable with respect to independat variable
r2 = lm.score(x,y)
print(r2)


# calculating mean absolute error and mean sruared error of model

from sklearn import metrics 
print('MAE:', metrics.mean_absolute_error(Y_test, predict))
print('MSE:', metrics.mean_squared_error(Y_test, predict))
print('RMSE:', np.sqrt(metrics.mean_squared_error(Y_test, predict)))

