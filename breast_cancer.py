# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 20:04:36 2018

@author: lenovo
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
import seaborn as sns

#load data
cancer = load_breast_cancer()

df_cancer = pd.DataFrame(np.c_[cancer['data'],cancer['target']],
                         columns=np.append(cancer['feature_names'],['target']))
#visulaizing the data
sns.pairplot(df_cancer,hue='target',vars=['mean radius','mean texture','mean perimeter','mean smoothness'])
sns.countplot(df_cancer['target'])
sns.heatmap(df_cancer.corr(),annot=True)

#load the data from the dataset
x = df_cancer.iloc[:,0:30].values
y = df_cancer.iloc[:,-1].values

#split the data 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)
#Preprocessing the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
 
#import machine learning model 
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

#train the model of the machine learning
svc = SVC()
svc.fit(x_train,y_train)

#predict the result 
y_pred = svc.predict(x_test)
cm = confusion_matrix(y_test,y_pred)
print(cm)

sns.heatmap(cm,annot=True)

print(classification_report(y_test,y_pred))
print("Accuracy of Model: "+str(svc.score(x_test,y_test)))










