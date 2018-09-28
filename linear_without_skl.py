
# Importing the necessary libraries


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# get_ipython().run_line_magic('matplotlib', 'inline')


# reading the dataset usinfg the pandas library
df = pd.read_csv("LungCapData.csv")


# information about the dataset 
df.info()
df.head()


# intializing the indepandent and depandant variables for linearing reression
y = df['Yearly Amount Spent'].values
x= df['Length of Membership'].values


# calculating the mean of x and y using numpy library
mean_x= np.mean(x)
mean_y = np.mean(y)


# checking the number of rows in x
n= len(x)



# In[43]:

numer = 0
denom = 0
for i in range(n):
    numer += (x[i]-mean_x)*(y[i]-mean_y)
   denom  += (x[i]-mean_x)**2

# calculating the value of y-intercept and slope  
b1 = numer/denom
b0 = mean_y- (b1* mean_x)

print (b1 , b0)    


## initializing sum of square of total
ss_t = 0


# initializing sum of square of residuals
ss_r = 0

# calculating the value of R square
for i in range(n):
    y_pred = b0+b1*x[i]
    ss_t += (y[i]-mean_y)**2
    ss_r += (y[i]-y_pred)**2
r2 = 1-(ss_r/ss_t)
print(r2)

