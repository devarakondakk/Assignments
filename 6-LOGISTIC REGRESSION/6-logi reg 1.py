# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 17:49:18 2023

@author: dekrk
"""

# STEP1 FRAME PROBLEM: client has subscribed term deposit or not Binomial y/n

# STEP2 COLECT READ DAT
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



X_cont = df[df.columns[[0,5,9,11,12,13,14]]]

X_cont
X_cate = df[df.columns[[1,2,3,4,6,7,8,10,15,16]]]
# LABEL ENCODING THE CATEGORICAL VARIABLES# label encoder
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
for column in bank.columns:
        if bank[column].dtype == np.number:
            continue
        bank[column] = LabelEncoder().fit_transform(bank[column])
bank.head()
df1.info()
# SPLITTINX X AND Y VARIABLES
x=df1.iloc[:,0:15]       
y=df1["y"]

# TARGET VARIABLE IS BINARY THUS WE USE LOGISTIC REGRESSION

from sklearn.linear_model import LogisticRegression
LOGI = LogisticRegression()
LOGI.fit(x,y)
y_pred = LOGI.predict(x)
y

y_pred_df=pd.DataFrame({'actual_y':y,'y_pred_prob':y_pred})
y_pred_df
# METRICS

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y,y_pred)
cm

TN = cm[0,0]
FP = cm[1,0]
TNR = TN/(TN+FP)
print("spcificity score:",TNR.round(3))
# WE GOT SPECIFITY SCORE OF 0.9 THAT 90% NEGATIVES ARE PREDICTED CORRECTLY
FN = cm[0,1]
TP = cm[1,1]
TPR = TP/(TP+FN)
print("PRECISION score:",TNR.round(3))
# WE GOT PRECISION SCORE OF 0.9 THAT 90% NEGATIVES ARE PREDICTED CORRECTLY


from sklearn.metrics import accuracy_score,f1_score
print("Accuracy score:",accuracy_score(y,y_pred).round(3))#Accuracy score: 0.887
print("f1 score score:",f1_score(y,y_pred).round(3))


y1 = y({'True': 1, 'False': 0}).astype(int)

# ROC Curve plotting and finding AUC value
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
fpr,tpr,thresholds=roc_curve(y1,LOGI.predict_proba(x)[:,1], pos_label=1)
plt.plot(fpr,tpr,color='red')
auc=roc_auc_score(y,y_pred)

plt.plot(fpr,tpr,color='red',label='logit model(area  = %0.2f)'%auc)
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
plt.ylabel('True Positive Rate')
plt.show()

print('auc accuracy:',auc)

