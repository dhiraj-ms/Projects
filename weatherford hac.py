
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
mainlog=pd.read_csv("MAMDAL_8_MAINLOG.csv")
mainlogtest=pd.read_csv("Mamdal3_WTLLIS_001_1.csv", low_memory = False)
hires=pd.read_csv("MAMDAL_8_HIRES.csv")
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.layers.convolutional import Convolution2D,MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils
from sklearn.cluster import KMeans


# In[2]:


print(mainlogtest)


# In[3]:


##mainlog=mainlog.drop(["MSPD","SPCG","GGCE","NPRL","DCOR","PDPE","HVOL","AVOL","TR21","TR11","TR22","TR12","BIT","DENN"],axis=1)
rowNames = ['DEPT','CLDC','DEN','DT35','FEFE','RILM','RILD']
mainlogcondensed = mainlog[rowNames]

print (mainlogcondensed)


# In[4]:


mainlogtestcondensed = mainlogtest[rowNames]
print (mainlogtestcondensed)


# In[5]:


for row in rowNames:
    indexNames = mainlogcondensed[ mainlogcondensed[row] == '-999.250' ].index
 
    # Delete these row indexes from dataFrame
    mainlogcondensed.drop(indexNames , inplace=True)


# In[6]:


print(mainlogcondensed)


# In[7]:


for row in rowNames:
    indexNames = mainlogtestcondensed[ mainlogtestcondensed[row] == '-999.250' ].index
 
    # Delete these row indexes from dataFrame
    mainlogtestcondensed.drop(indexNames , inplace=True)


# In[8]:


print(mainlogtestcondensed)


# In[9]:


for row in rowNames:
    indexNames = mainlogtestcondensed[ mainlogtestcondensed[row] == '-9999' ].index
 
    # Delete these row indexes from dataFrame
    mainlogtestcondensed.drop(indexNames , inplace=True)


# In[10]:


print(mainlogtestcondensed)


# In[17]:


x_train = mainlogcondensed
x_test = mainlogtestcondensed


# In[18]:


print(x_train.shape)
print(x_test.shape)


# In[19]:


n_clusters = 5


# In[20]:


x_train = x_train.drop(x_train.index[[0, 1]])


# In[21]:


x_test = x_test.drop(x_test.index[[0, 1]])


# In[22]:


kmeans = KMeans(n_clusters=5, random_state=0).fit(x_train)


# In[25]:


y_kmeans = kmeans.predict(x_test)


# In[27]:


plt.scatter(x_test[:, 2], x_test[:, 4], c=y_kmeans, s=50, cmap='viridis')

