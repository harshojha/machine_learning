
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('911.csv')


# In[3]:


df.info()


# In[4]:


df['zip'].value_counts().head(5)


# In[5]:


df['title'].nunique()


# In[6]:


df['Reason'] = df['title'].apply(lambda title: title.split(':')[0])


# In[7]:


df['Reason'].value_counts()


# In[8]:


type(df['timeStamp'].iloc[0])


# In[9]:


df['timeStamp'] = pd.to_datetime(df['timeStamp'])


# In[10]:


df['Hour'] = df['timeStamp'].apply(lambda time: time.hour)
df['Month'] = df['timeStamp'].apply(lambda time: time.month)
df['Day of Week'] = df['timeStamp'].apply(lambda time: time.dayofweek)


# In[11]:


dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}


# In[12]:


df['Day of Week'] = df['Day of Week'].map(dmap)


# In[13]:


sns.countplot(x='Day of Week',data=df,hue='Reason',palette='viridis')


# In[14]:


byMonth = df.groupby('Month').count()
byMonth.head()


# In[15]:


byMonth['twp'].plot()


# In[16]:


sns.lmplot(x='Month',y='twp',data=byMonth.reset_index())


# In[17]:


df['Date']=df['timeStamp'].apply(lambda t: t.date())


# In[18]:


df.groupby('Date').count()['twp'].plot()
plt.tight_layout()


# In[19]:


df[df['Reason']=='Traffic'].groupby('Date').count()['twp'].plot()
plt.title('Traffic')
plt.tight_layout()


# In[20]:


df[df['Reason']=='EMS'].groupby('Date').count()['twp'].plot()
plt.title('EMS')
plt.tight_layout()


# In[21]:


dayHour = df.groupby(by=['Day of Week','Hour']).count()['Reason'].unstack()
dayHour.head()


# In[22]:


plt.figure(figsize=(12,6))
sns.heatmap(dayHour,cmap='viridis')


# In[23]:


sns.clustermap(dayHour,cmap='viridis')

