import argparse
import flwr as fl
import pandas as pd
import numpy as np
import utils
import tensorflow as tf
import tensorflow.keras as keras
import torch
from flwr.common import (NDArrays,ParametersRes, FitIns, EvaluateRes, EvaluateIns)
import timeit

# Load dataset
normal_df = pd.read_csv('datasets/ptbdb_normal.csv', header=None)
abnormal_df = pd.read_csv('datasets/ptbdb_abnormal.csv', header=None)
#df = pd.read_csv('datasets/mitbih_test.csv', header=None, delimiter=',',nrows=10000) 
#print(df.shape)
#print(df.head())

#print(normal_df.shape)
#print(abnormal_df.shape)

#print(normal_df.isnull().sum())
#print(abnormal_df.isnull().sum())

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold


#print(X_train.shape)
#print(X_test.shape)
#print(y_train.shape)
#print(y_test.shape)

print(X_train[0])
print(y_train[0])

model = LogisticRegression(random_state=0, max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)  
print(classification_report(y_test, y_pred))
print('Accuracy of test data : '+accuracy_score(y_test, y_pred).astype(str))
#print('Accuracy of train data : '+model.accuracy(X_train, y_train))
#model = LogisticRegression(random_state=0, max_iter=1000)
df = pd.concat([normal_df, abnormal_df], axis=0, ignore_index=True)
print(df.shape)
#Shuffle the data
df = df.sample(df.shape[0], random_state=42)
#X values should be until row 187
X = df.iloc[:, :186].values
#Y values should be from row 187
y = df.iloc[:, 187].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)