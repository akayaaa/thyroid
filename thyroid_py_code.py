#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing datasets
dataset = pd.read_csv('thyroid.csv')

print(dataset)

x = dataset.iloc[:,1:5].values
y = dataset.iloc[:,5:].values
print(y)

#train test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=0)

#scaling datasets
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)

#prediction with logistic regression
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(x_train,y_train)

y_pred=logr.predict(x_test)
#prediction part
print(y_pred)
#control part
print(y_test)

x_pred=logr.predict(x_train)
#prediction part
print(x_pred)
#control part
print(y_train)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)
#knn 
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5,metric='minkowski')
knn.fit(x_train,y_train)

y_pred = knn.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)

cm=confusion_matrix(y_train, x_pred)  
print(cm)  
