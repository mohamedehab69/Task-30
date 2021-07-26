#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns

sns.set(rc={'figure.figsize': [10, 10]}, font_scale=1.2)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('Country-data.csv')
df


# In[3]:


df.drop('country',axis=1,inplace=True)
df


# In[4]:


from sklearn.preprocessing import StandardScaler

scaler= StandardScaler()

df_scaled= scaler.fit_transform(df)


# In[5]:


scores = []
for i in range(1, 50):
    model = KMeans(n_clusters=i)
    model.fit(df_scaled)
    scores.append(model.inertia_)
plt.plot(range(1, 50), scores)
plt.title("Elbow Method")
plt.xlabel("num of clusters")
plt.ylabel("Score")


# In[6]:


model= KMeans(n_clusters=25)
clusters = model.fit_predict(df_scaled)
clusters


# In[7]:


df['clusters']=clusters
df


# In[8]:


data=[90.2,10.0,7.58,44.9,1610,9.44,56.2,5.82,553]


# In[9]:


model.predict(scaler.transform([data]))


# In[ ]:




