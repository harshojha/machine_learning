
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train = pd.read_csv("titanic_train.csv")


# In[3]:


train.head()


# In[4]:


train.isnull()


# In[7]:


train.drop('Cabin', axis=1, inplace=True)


# In[8]:


train.head()


# In[14]:


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


# In[15]:


train["Age"]= train[["Age", "Pclass"]].apply(impute_age, axis= 1)


# In[16]:


train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)


# In[17]:


train.isnull()


# In[24]:


sex = pd.get_dummies(train["Sex"], drop_first= True)


# In[25]:


embark = pd.get_dummies(train["Embarked"], drop_first= True)


# In[26]:


pd.get_dummies(train["Embarked"], drop_first= True)


# In[29]:


train.drop(["Sex", "Embarked", "Name", "PassengerId", "Ticket"], axis=1, inplace= True)


# In[30]:


train.head()


# In[34]:


train= pd.concat([train, sex, embark],axis=1)


# In[35]:


train.head()


# In[36]:


from sklearn.model_selection import train_test_split


# In[39]:


X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1),train['Survived'], test_size=0.30, random_state=101)


# In[40]:


from sklearn.linear_model import LogisticRegression
logmodel= LogisticRegression()
logmodel.fit(X_train, y_train)


# In[41]:


predictions = logmodel.predict(X_test)


# In[42]:


from sklearn.metrics import classification_report


# In[43]:


print(classification_report(y_test,predictions))

