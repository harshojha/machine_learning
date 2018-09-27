
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.datasets import fetch_20newsgroups
data = fetch_20newsgroups()
data.target_names


# In[3]:


categories = ['alt.atheism',
 'comp.graphics',
 'comp.os.ms-windows.misc',
 'comp.sys.ibm.pc.hardware',
 'comp.sys.mac.hardware',
 'comp.windows.x',
 'misc.forsale',
 'rec.autos',
 'rec.motorcycles',
 'rec.sport.baseball',
 'rec.sport.hockey',
 'sci.crypt',
 'sci.electronics',
 'sci.med',
 'sci.space',
 'soc.religion.christian',
 'talk.politics.guns',
 'talk.politics.mideast',
 'talk.politics.misc',
 'talk.religion.misc']


# In[4]:


## Training the data on these categories

train = fetch_20newsgroups(subset = "train", categories= categories)

## testing the data for these categories

test = fetch_20newsgroups(subset = 'test', categories= categories)


# In[6]:


## printing training data 

print (len(train.data[5]))


# In[9]:


from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import make_pipeline


# In[10]:


## creating a model based on multiniminal naive_bayes

model = make_pipeline(TfidfVectorizer(), MultinomialNB())

## training the model with the train data

model.fit(train.data, train.target)

## creating labels for the test

labels = model.predict(test.data)


# In[12]:


## confusion matrics

from sklearn.metrics import confusion_matrix

mat = confusion_matrix(test.target, labels)

sns.heatmap(mat.T , square= True, annot= True, fmt= "d", cbar= False, xticklabels= train.target_names, yticklabels= train.target_names)


plt.xlabel("true label")
plt.ylabel("predicted label");

