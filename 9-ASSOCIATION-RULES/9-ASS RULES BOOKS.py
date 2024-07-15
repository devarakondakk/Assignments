# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 00:09:11 2023

@author: dekrk
"""
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
df=pd.read_csv('book.csv')
df

#--1.Association rules with 10% Support and 70% confidence
from mlxtend.frequent_patterns import apriori,association_rules
freq_itemsets=apriori(df,min_support=0.1,use_colnames=True)
freq_itemsets

rules=association_rules(freq_itemsets,metric='lift',min_threshold=0.7)
rules

# Lift Ratio > 1 is a good influential rule in selecting the associated transactions
rules.sort_values('lift',ascending=False)
rules[rules.lift>1]


# visualization of obtained rule
import matplotlib.pyplot as plt
plt.scatter(rules['support'],rules['confidence'])
plt.xlabel('support')
plt.ylabel('confidence') 
plt.show()


# --2. Association rules with 20% Support and 60% confidence
freq_itemsets2=apriori(df,min_support=0.2,use_colnames=True)
freq_itemsets2

rules2=association_rules(freq_itemsets2,metric='lift',min_threshold=0.6)
rules2
rules2.sort_values('lift',ascending=False)
rules2[rules2.lift>1]


# visualization of obtained rule
import matplotlib.pyplot as plt
plt.scatter(rules2['support'],rules2['confidence'])
plt.xlabel('support')
plt.ylabel('confidence') 
plt.show()


# --3. Association rules with 5% Support and 80% confidence
freq_itemsets3=apriori(df,min_support=0.05,use_colnames=True)
freq_itemsets3

rules3=association_rules(freq_itemsets3,metric='lift',min_threshold=0.8)
rules3
rules3.sort_values('lift',ascending=False)
rules3[rules3.lift>1]


# visualization of obtained rule
import matplotlib.pyplot as plt
plt.scatter(rules3['support'],rules3['confidence'])
plt.xlabel('support')
plt.ylabel('confidence') 
plt.show()