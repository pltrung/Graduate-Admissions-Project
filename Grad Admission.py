# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 14:51:10 2019

@author: pltru
"""

#importing libraries

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys
import os 

df = pd.read_csv('Google Drive\\Masters\\Graduate Admissions Project\\Admission_Predict.csv', sep = ",")

df.info()
df.head()
df.corr()
df.describe()
df=df.rename(columns = {'Chance of Admit ':'Chance of Admit'})
#correlation plot 
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(df.corr(), ax = ax, annot = True, linewidths = 0.05, fmt = '.2f', cmap = 'magma')
plt.show()

#data visualization
y = np.array([len(df[df.Research == 0]), len(df[df.Research == 1])])
x = ["No Research", "Yes Research"]

#barplot
plt.bar(x,y)
plt.title("Research Comparison")
plt.ylabel("Frequency of Candidates")
plt.show()

#TOEFL
y = np.array([df["TOEFL Score"].min(),df["TOEFL Score"].mean(),df["TOEFL Score"].max()])
x = ["Worst","Average","Best"]
plt.bar(x,y)
plt.title("TOEFL Scores")
plt.xlabel("Level")
plt.ylabel("TOEFL Score")
plt.show()

#GRE
df["GRE Score"].plot(kind = 'hist',bins = 200,figsize = (6,6))
plt.title("GRE Scores")
plt.xlabel("GRE Score")
plt.ylabel("Frequency")
plt.show()

#University Rating vs CGPA
plt.scatter(df["University Rating"],df['CGPA'])
plt.title("CGPA Scores for University Ratings")
plt.xlabel("University Rating")
plt.ylabel("CGPA")
plt.show()

#GCPA vs GRE
plt.scatter(df["GRE Score"],df["CGPA"])
plt.title("CGPA for GRE Scores")
plt.xlabel("GRE Score")
plt.ylabel("CGPA")
plt.show()

#ranking vs acceptance rate
s = df[df["Chance of Admit"] >= 0.75]["University Rating"].value_counts()
plt.title("University Ratings of Candidates with an 75% acceptance chance")
s.plot(kind='bar',figsize=(20, 20))
plt.xlabel("University Rating")
plt.ylabel("Candidates")
plt.show()

#preparing for regression

serialno = df["Serial No."].values
df.drop(["Serial No."], axis = 1, inplace = True)

y = df["Chance of Admit"].values
x = df.drop(["Chance of Admit"], axis = 1)

#data partition
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.20, random_state = 42)

#random forest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 100, random_state = 42)
rf.fit(x_train,y_train)
y_head_rf = rf.predict(x_test) 

#result
from sklearn.metrics import r2_score
y_rf_train_pred = rf.predict(x_train)
print("R-squared score (train dataset):", r2_score(y_train, y_rf_train_pred))
print("R_square score (test dataset): ", r2_score(y_test,y_head_rf))

#classification, if Chance of Admit is greater than 0.75, the candidate will recieve 1 label (meaning accepted)

#normalization
from sklearn.preprocessing import MinMaxScaler
scalerX = MinMaxScaler(feature_range=(0, 1))
x_train[x_train.columns] = scalerX.fit_transform(x_train[x_train.columns])
x_test[x_test.columns] = scalerX.transform(x_test[x_test.columns])

y_train_01 = [1 if each > 0.75 else 0 for each in y_train]
y_test_01  = [1 if each > 0.75 else 0 for each in y_test]

y_train_01 = np.array(y_train_01)
y_test_01 = np.array(y_test_01)


#logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score

lr = LogisticRegression()
lr.fit(x_train,y_train_01)

#results

print("precision_score: ", precision_score(y_test_01,lr.predict(x_test)))
print("recall_score: ", recall_score(y_test_01,lr.predict(x_test)))
print("f1_score: ",f1_score(y_test_01,lr.predict(x_test)))
print("score: ", lr.score(x_test,y_test_01))

#confusion matrix
from sklearn.metrics import confusion_matrix
cm_lr = confusion_matrix(y_test_01,lr.predict(x_test))

f, ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm_lr,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.title("Test for Test Dataset")
plt.xlabel("predicted y values")
plt.ylabel("real y values")
plt.show()

#random forest classification
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 100,random_state = 1)
rfc.fit(x_train,y_train_01)

#results
print("score: ", rfc.score(x_test,y_test_01))
print("precision score: ", precision_score(y_test_01, rfc.predict(x_test)))
print("recall score: ", recall_score(y_test_01, rfc.predict(x_test)))
print("f1 score: ", f1_score(y_test_01, rfc.predict(x_test)))

#confusion matrix
cm_rfc = confusion_matrix(y_test_01,rfc.predict(x_test))

f, ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm_rfc,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.title("Test for Test Dataset")
plt.xlabel("predicted y values")
plt.ylabel("real y values")
plt.show()




