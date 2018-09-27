
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


df = pd.read_csv("LungCapData.csv")


# In[6]:


df.info()


# In[7]:


df.head()


# In[12]:


y = df['Yearly Amount Spent'].values


# In[13]:


y


# In[39]:


x


# In[40]:


mean_x= np.mean(x)
mean_y = np.mean(y)


# In[41]:


n= len(x)


# In[42]:


n


# In[43]:


numer = 0
denom = 0
for i in range(n):
    numer += (x[i]-mean_x)*(y[i]-mean_y)
    denom  += (x[i]-mean_x)**2

b1 = numer/denom
b0 = mean_y- (b1* mean_x)

print (b1 , b0)    


# In[44]:


## print cofficient

ss_t = 0
ss_r = 0

for i in range(n):
    y_pred = b0+b1*x[i]
    ss_t += (y[i]-mean_y)**2
    ss_r += (y[i]-y_pred)**2
r2 = 1-(ss_r/ss_t)
print(r2)

