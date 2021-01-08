#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.neighbors import NearestCentroid
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns


# In[3]:


df = pd.read_table('C:/Users/Mahsa/Desktop/tst/expanded.txt', delimiter=',', header=None)
column_labels = [
    'class', 'cap-shape', 'cap-surface', 'cap-color', 'bruised', 'odor',
    'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
    'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
    'stalk-surface-below-ring', 'stalk-color-above-ring',
    'stalk-color below-ring', 'veil-type', 'veil-color', 'ring-number',
    'ring-type', 'spore-print-color', 'population', 'habitat']


# In[4]:


df.columns = column_labels
df = df[df['stalk-root'] != '?']
X = df.loc[:, df.columns != 'class']
y = df['class'].to_frame()
y['class'].value_counts()

X_enc = pd.get_dummies(X)

scaler = StandardScaler()
X_std = scaler.fit_transform(X_enc)
le = LabelEncoder()
y_enc = le.fit_transform(y.values.ravel())
X_train, X_test, y_train, y_test = train_test_split(
    X_std,
    y_enc,
    test_size=0.2,
    stratify=y_enc,
    random_state=42
)


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
KNeighborsClassifier(algorithm='auto',
                     leaf_size=30,
                     metric='minkowski',
                     metric_params=None,
                     n_jobs=1,
                     n_neighbors=5,
                     p=2,
                     weights='uniform')
pred = knn.predict(X_test)
print("Predictions form the classifier:")
print(pred)
print("Predictions form the classifier:")
print(y_test)
np.save('sometext0', pred)
np.save('sometext1', y_test)
accuracy_score(y_test, pred)


# In[ ]:




