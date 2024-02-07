#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle


# ## Importing Data as Dataframe(df)

# In[2]:


column_names = ['id', 'diagnosis', 'radius_mean', 'texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concave_points_mean','symmetry_mean','fractal_dimension_mean','radius_se','texture_se','perimeter_se','area_se','smoothness_se','compactness_se','concavity_se','concave_points_se','symmetry_se','fractal_dimension_se','radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst','compactness_worst','concavity_worst','concave_points_worst','symmetry_worst','fractal_dimension_worst']
df = pd.read_csv('breast+cancer+wisconsin+diagnostic/wdbc.data', names=column_names)


#  ## Printing the first 5 rows

# In[3]:


df.head()


# ## Seperating Data and Labels

# In[4]:


X = df.drop('diagnosis', axis=1)
y = df['diagnosis']


# ## Splitting Data in training and testing sets

# In[5]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# ## Model Training using Random Forest Classifier(RFC)

# In[6]:


rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)


# ## Model Evaluation

# In[7]:


y_pred = rfc.predict(X_test)


# In[8]:


accuracy_score(y_test, y_pred)


# ## Saving the model

# In[10]:


pickle.dump(rfc, open('banish_model_v2.pkl', 'wb'))


# In[ ]:




