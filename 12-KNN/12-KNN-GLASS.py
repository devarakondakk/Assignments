# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 16:25:02 2023

@author: dekrk
"""
## Import data file
import pandas as pd
df = pd.read_csv('glass.csv')
df
df.info()
df.shape
df.describe()
df[df.duplicated()].shape
df[df.duplicated()]
df1 = df.drop_duplicates()
df1
import seaborn as sns
sns.heatmap(df.corr())
import matplotlib.pyplot as plt
sns.pairplot(df1,hue='Type')
plt.show()

plt.figure(figsize=(10,10))
sns.boxplot(data=df1, orient="h");
from dataprep.eda import plot
plot(df1)


