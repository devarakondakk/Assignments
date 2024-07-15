# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 15:39:07 2023

@author: dekrk
"""
# Import libraries
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', -1)
# import the file
df=pd.read_csv('wine.csv')
df.head()

# ------descriptive stats
df['Type'].value_counts()
# three types of wine present thus drop from analysis
df1=df.drop('Type',axis=1)
df1.head()
df1.info()
df1.describe()
# check for missing values
df1.isnull().sum().value_counts #no null values found
df1.duplicated().sum() 
df1[df1.duplicated()]#no duplicates found

# ----EDA----
df.skew()
import seaborn as sns
sns.set(style='dark',font_scale=1.3, rc={'figure.figsize':(20,20)})
ax=df1.hist(bins=20,color='blue' )

import matplotlib.pyplot as plt
df1.boxplot()
df1.plot( kind = 'box', subplots = True, layout = (4,4), sharex = False, sharey = False,color='black')
plt.show()
sns.pairplot(df1)

# correlation heatmap
plt.figure(figsize=(18,12))
sns.heatmap(df1.corr(), annot=True, linewidths =.5, fmt ='.1f',cmap="coolwarm")
plt.show()

# data preprocessing
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
df2 = ss.fit_transform(df1)
df2.shape

# ----PCA
from sklearn.decomposition import PCA
pca = PCA()
pc = pca.fit_transform(df2)
pc

vrnce=pca.explained_variance_ratio_ # amount of variance that each PCA has
sum(pca.explained_variance_ratio_)
import numpy as np
vrnce2=np.cumsum(np.round(vrnce,4)*100)# Cummulative variance of each PCA
vrnce2

# Variance plot for PCA components obtained 
plt.plot(vrnce2,color='magenta')
pd.DataFrame(pc)

# Final Dataframe using 3 principal component scores 
df3=pd.concat([df['Type'],pd.DataFrame(pc[:,0:3],columns=['PC1','PC2','PC3'])],axis=1)
df3

# Visualization of PCAs
fig=plt.figure(figsize=(16,12))
sns.scatterplot(data=df3)

# ---------Checking with other Clustering Algorithms
# -------1  Hierarchical Clustering
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

plt.figure(figsize=(10,8))
dendrogram=sch.dendrogram(sch.linkage(df2,'complete'))

# crating clusters=Y 3 clusters
hclust=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='ward')
hclust

Y=pd.DataFrame(hclust.fit_predict(df2),columns=['clustersid'])
Y['clustersid'].value_counts()

# Adding clusters to dataset
df4=df.copy()
df4['clustersid']=hclust.labels_
df4

# ------2 k-mean cluster----

from sklearn.cluster import KMeans
wcss=[]
for i in range (1,6):
    kmeans=KMeans(n_clusters=i,random_state=2)
    kmeans.fit(df2)
    wcss.append(kmeans.inertia_)
    
# Plotting K values vs WCSS for Elbow graph to choose K (no. of clusters)
plt.plot(range(1,6),wcss)   
plt.title('Elbow Graph')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Cluster algorithm using K=3
kclust=KMeans(3,random_state=30).fit(df2)
kclust

kclust.labels_
# Assign K-clusters to the data set
df5=df.copy()
df5['clusters3id']=kclust.labels_
df5

df5['clusters3id'].value_counts()
'''
Observation:
    The 3 clusters has been clustered but has a negliglable amount of difference compared to original classified Feature
    We have perfectly clustered the data into Three Types as compared to classification of three types of Wine was indicated in the Original Dataset in 'Type' Column

'''