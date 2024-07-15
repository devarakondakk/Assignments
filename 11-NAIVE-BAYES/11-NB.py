# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 13:36:47 2023

@author: dekrk
"""
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 0)

s_train = pd.read_csv('SalaryData_Train.csv')
s_train.head()

s_test = pd.read_csv('SalaryData_Test.csv')
s_test.head()

# ----EDA----
list(s_train)
s_train.shape
s_train.info()
#categ variables (9)= workclass, education,maraital,occu,relation,race,sex,nativ,salary
# traget variable is the salalry
s_train.describe()
s_train.isnull().sum()#no null values
s_train.boxplot() #many outliers present, cannot remove them
s_train[s_train.duplicated()]#3258 duplicates found
s_train =s_train.drop_duplicates()



s_test.shape
s_test.info() 
s_test.describe()
s_test.isnull().sum()#no null values
s_test.boxplot()
s_test[s_test.duplicated()]#930 duplicates found
s_test =s_test.drop_duplicates()

# ----------Data transofrmation
from sklearn.preprocessing import LabelEncoder

categ = ['age', 'workclass', 'education', 'educationno', 'maritalstatus',
       'occupation', 'relationship', 'race', 'sex', 'capitalgain',
       'capitalloss', 'hoursperweek', 'native', 'Salary']
LE = LabelEncoder()
for i in categ:
        s_train[i]= LE.fit_transform(s_train[i])
    
s_train.head()

for i in categ:
        s_test[i]= LE.fit_transform(s_test[i])
    
s_test.head()
s_test

# storing the values in x_train,y_train,x_test & y_test for spliting the data in train and test 
X_train = s_train[categ[0:13]].values
Y_train = s_train[categ[13]].values
X_test = s_test[categ[0:13]].values
Y_test = s_test[categ[13]].values


from sklearn.naive_bayes import MultinomialNB
NB = MultinomialNB()

NB.fit(X_train,Y_train)

Y_pred_train = NB.predict(X_train)
Y_pred_test = NB.predict(X_test)

from sklearn.metrics import accuracy_score , recall_score, confusion_matrix
print("TRAIN Accuracy score: " ,accuracy_score(Y_train,Y_pred_train).round(2))
# TRAIN Accuracy score:  0.78
print("TEST Accuracy score: " ,accuracy_score(Y_test,Y_pred_test).round(2))
# TEST Accuracy score:  0.78

recall_score(Y_train,Y_pred_train).round(2)#0.33
confusion_matrix(Y_train,Y_pred_train)

from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred_test))
