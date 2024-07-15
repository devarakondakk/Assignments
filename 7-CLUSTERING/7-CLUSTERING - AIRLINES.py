# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 16:29:14 2023

@author: dekrk
"""
import pandas as pd


pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

df = pd.read_excel('EastWestAirlines.xlsx',sheet_name='data')

# renaming columns with special charecters
df.rename(columns={'ID#':'ID', 'Award?':'Award'}, inplace=True)
# setting index id as ID
df.set_index('ID',inplace=True)
df

df.describe()
df.info() #all values are  continous
df.isnull().sum().value_counts #NO NULL VALUES
df.duplicated().sum()#NO DUPLICATE VALUES

# -----Exploratory Data Analysis-------
import matplotlib.pyplot as plt
for feature in df.columns:
    data=df.copy()
    data[feature].hist(bins=25)
    plt.ylabel('Count')
    plt.title(feature)
    plt.show()

import seaborn as sns
plt.figure(figsize=(12,8))
sns.boxplot(data=df)# many outlier found in balennce and bonus miles

# using squre root of data to accurately find outliers
import numpy as np
plt.figure(figsize=(12,8))
sns.boxplot(data=np.sqrt(df))
#we also see outliers in flight miles, 12 months flight miles, days since enroll

# -------Data Visualization----------
# Corelation Analysis
plt.figure(figsize=(18,12))
sns.heatmap(df.corr(), annot=True, linewidths =.5, fmt ='.1f',cmap="coolwarm")
plt.show()


# ------kmeans model building------------

from sklearn.cluster import KMeans
kmeans=KMeans().fit(df)
score=[]
K=range(1,20)

for i in K:
    kmeans=KMeans(n_clusters=i,init="k-means++",random_state=3,n_init='auto')
    kmeans.fit(df)
    score.append(kmeans.inertia_)

#visualize;
import matplotlib.pyplot as plt
plt.plot(K,score,color="red")
plt.xlabel("k value")
plt.ylabel("wcss value") # within-cluster sum-of-squares criterion 
plt.show()

# number of clusters 3-4 selectable

# for K-elbow;

from yellowbrick.cluster import KElbowVisualizer
visualizer=KElbowVisualizer(kmeans,k=(1,20),n_init='auto')
visualizer.fit(df)
visualizer.poof()
plt.show()

# 4 clusters to be divided into

#final model;
kmeans=KMeans(n_clusters=4,init="k-means++",n_init='auto').fit(df)

# STANDARDIZE DATA

from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
df2=SS.fit_transform(df)
df2.shape

visualizer=KElbowVisualizer(kmeans,k=(1,20),n_init='auto')
visualizer.fit(df2)
visualizer.poof()
plt.show()

# ----FINAL MODEL---
kmeans=KMeans(n_clusters=4,init="k-means++",n_init='auto').fit(df2)
#add tag values;
cluster=kmeans.labels_
cluster

# ADD COLUMMN
df2["cluster_no"]=cluster
df2.head()


import scipy.cluster.hierarchy as sch

dendogram=sch.dendrogram(sch.linkage(df,method="ward"))
plt.show()
