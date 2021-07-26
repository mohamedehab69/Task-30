#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns

sns.set(rc={'figure.figsize': [7, 7]}, font_scale=1.2)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[25]:


df = pd.read_csv('university.csv')


# In[26]:


df


# In[27]:


df= pd.get_dummies(df,columns=['Private'],drop_first=True)
df


# In[28]:


df.drop('University',axis=1,inplace=True)
df


# In[31]:


from sklearn.preprocessing import StandardScaler

scaler= StandardScaler()

df_scaled= scaler.fit_transform(df)


# In[32]:


model= KMeans(n_clusters=3)
clusters = model.fit_predict(df_scaled)
clusters


# In[33]:


df['clusters'] = clusters
df


# In[34]:


scores = []
for i in range(1, 50):
    model = KMeans(n_clusters=i)
    model.fit(df_scaled)
    scores.append(model.inertia_)
plt.plot(range(1, 50), scores)
plt.title("Elbow Method")
plt.xlabel("num of clusters")
plt.ylabel("Score")


# In[35]:


model= KMeans(n_clusters=30)
clusters = model.fit_predict(df_scaled)
clusters


# In[37]:


df['clusters']=clusters
df


# In[41]:


data=[1660,1232,700,25,52,2900,547,7440,3300,450,2200,70,78,18.1,12,7041,60,1]


# In[42]:


model.predict(scaler.transform([data]))


# In[ ]:




