# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 00:24:43 2023

@author: dekrk
"""
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
df=pd.read_csv('book.csv',encoding='Latin1')
df

df2=df.iloc[:,1:]
df2
df2.sort_values(['User.ID'])


# number of unique users in the dataset
len(df2['User.ID'].unique())

# number of unique books in the dataset
len(df2['Book.Title'].unique())

# converting long data into wide data using pivot table
df3=df2.pivot_table(index='User.ID',columns='Book.Title',values='Book.Rating').reset_index(drop=True)
df3

# Impute those NaNs with 0 values
df3.fillna(0,inplace=True)
df3

# Calculating Cosine Similarity between Users on array data
from sklearn.metrics import pairwise_distances
reader_sim=1-pairwise_distances(df3.values,metric='cosine')
reader_sim
reader_sim2=pd.DataFrame(reader_sim)
reader_sim2.index=df2['User.ID'].unique()
reader_sim2.columns=df2['User.ID'].unique()
reader_sim2

# Nullifying diagonal values
import numpy as np
np.fill_diagonal(reader_sim,0)
reader_sim2


# Most Similar Users
reader_sim2.idxmax(axis=1)



# extract the books which userId 276744 & 276726 have watched
df2[(df2['User.ID']==276744) | (df2['User.ID']==276726)]


user_1=df2[(df2['User.ID']==276744)]
user_2=df2[(df2['User.ID']==276726)]



user_1['Book.Title']
user_2['Book.Title']




