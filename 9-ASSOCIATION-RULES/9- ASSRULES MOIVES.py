# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 23:51:54 2023

@author: dekrk
"""
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
df=pd.read_csv('my_movies.csv')
df

# ----descriptive stats
df.shape
df.info()


df2=df.iloc[:,5:]
df2


# -----EDA
import matplotlib.pyplot as plt
from wordcloud import WordCloud

plt.rcParams['figure.figsize'] = (15, 15)
wordcloud = WordCloud(background_color = 'white', width = 1200,  height = 1200, max_words = 121).generate(str(df2.sum()))
plt.imshow(wordcloud)
plt.axis('off')
plt.title('Items',fontsize = 20)
plt.show()

plt.figure(figsize = (12,8))
plt.pie(df2.sum(),labels=df2.columns)
plt.title("Movies wrt Purchase Rate")
plt.show()
# -----affinity analysis
from mlxtend.frequent_patterns import apriori,association_rules
repitems=apriori(df2,min_support=0.1,use_colnames=True)# with 10% support
repitems


assrules=association_rules(repitems,metric='lift',min_threshold=0.7)# 70% confidence
assrules



# Lift Ratio > 1 is a good influential rule in selecting the associated transactions
assrules[assrules.lift>1]


# visualization of obtained rule

plt.scatter(assrules['support'],assrules['confidence'])
plt.xlabel('support')
plt.ylabel('confidence')
plt.show()

# with 5% support, 90% confidence
repitems2=apriori(df2,min_support=0.05,use_colnames=True)
repitems2

# 90% confidence
assrules2=association_rules(repitems2,metric='lift',min_threshold=0.9)
assrules2

# Lift Ratio > 1 is a good influential rule in selecting the associated transactions
assrules2[assrules2.lift>1]


# visualization of obtained rule
plt.scatter(assrules2['support'],assrules2['confidence'])
plt.xlabel('support')
plt.ylabel('confidence')
plt.show()


