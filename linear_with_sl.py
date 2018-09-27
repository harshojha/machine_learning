
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[12]:


df = pd.read_csv("Ecommerce Customers")


# In[15]:


df.describe()


# In[20]:


x= df['Length of Membership'].values


# In[24]:


n= len(x)


# In[25]:


x= x.reshape((n,1))


# In[26]:


y = df['Yearly Amount Spent'].values


# In[27]:


from sklearn.model_selection import train_test_split


# In[29]:


X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.4, random_state=101)


# In[30]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,Y_train)


# In[31]:


print(lm.intercept_)


# In[34]:


predict = lm.predict(X_test)


# In[35]:


plt.plot(predict)


# In[36]:


plt.scatter(Y_test, predict)


# In[37]:


r2 = lm.score(x,y)


# In[38]:


print(r2)


# In[35]:


from sklearn import metrics


# In[36]:


print('MAE:', metrics.mean_absolute_error(Y_test, predict))
print('MSE:', metrics.mean_squared_error(Y_test, predict))
print('RMSE:', np.sqrt(metrics.mean_squared_error(Y_test, predict)))

