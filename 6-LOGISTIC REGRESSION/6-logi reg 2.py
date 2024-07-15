# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 20:55:18 2023

@author: dekrk
"""
import numpy as np
import pandas as pd
df = pd.read_csv('bank-full.csv', delimiter=';')
df
df.shape
df.head()



df.describe()
df.info()

df.isnull().sum() #NO MISSING VALUES
df[df.duplicated()] #NO DUPLICATES FOUND
df.corr()
df.boxplot('duration',vert=False)#many outliers found

#Removing the outliers using quartile method 
q3 = df['duration'].quantile(.75)
q1 = df['duration'].quantile(.25)
iqr = q3-q1
iqr
upperrange = q3+1.5*iqr
bottomrange = q1-1.5*iqr
df1 = df[(df['duration']>bottomrange) & (df['duration']<upperrange)] 
df1.boxplot('duration',vert=False)


# LABEL ENCODING THE CATEGORICAL VARIABLES# label encoder
from sklearn.preprocessing import LabelEncoder

for column in df.columns:
        if df[column].dtype == np.number:
            continue
        df[column] = LabelEncoder().fit_transform(df[column])
df.head()


#correlation visualization
import matplotlib.pyplot as plt
import seaborn as sns
plt.subplots(figsize=(8,8))
sns.heatmap(df.corr(), annot = True, fmt = '0.0%')



#split data into 70% training and 30% testing.
from sklearn.model_selection import train_test_split
X = df.iloc[:,0:15].values
Y = df.iloc[:,0].values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3, random_state = 1)
X_train

#Logistic Regression model building
from sklearn.linear_model import LogisticRegression
LOGI=LogisticRegression()
# the option chosen is ‘ovr’, then a binary problem is fit for each label. 
LOGI.fit(X_train,Y_train)

#Training Model Score
LOGI.score(X_train,Y_train)


#Testing Model Score
LOGI.score(X_test,Y_test)