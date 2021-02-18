import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt  
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split


##Step 1: Get Data from CSV

df = pd.read_csv("csv/dataset.csv")
##Step 2: Seperate Labels and Features
Y = df["label"]
X = df.drop(["label"],axis=1) 


##Step 3: Make sure you have the correct Feature / label combination in training
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=38)


##Step 4: Build a Model and Save it
model = svm.SVC()
model.fit(x_train,y_train)

joblib.dump(model,"model/my_Svmwith4labels")

print("ACcuracy",model.score(x_test,y_test))
 

##Step5 : Print Accuracy 
