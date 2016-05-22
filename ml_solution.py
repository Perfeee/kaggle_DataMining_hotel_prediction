#!/usr/bin/env python
# coding=utf-8

import numpy as np
import pandas as pd
import sklearn.preprocessing as pre
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

rows = 1000000
train = pd.read_csv("./train.csv",usecols=["user_location_country","user_location_region","user_location_city","hotel_continent","hotel_country","hotel_market","hotel_cluster"],nrows=rows)
train = np.array(train)

train[:,:-1] = pre.MinMaxScaler(feature_range=(0,3000)).fit_transform(X=train[:,:6])
print(train[0:5,:])
train_data = train[0:-1000,:]
test_data = train[-1000:,:]


dtc = DecisionTreeClassifier()
dtc.fit(X=train_data[:,0:-1],y=train_data[:,-1])
print("DecisionTreeClassifier_Score:",dtc.score(X=test_data[:,0:-1],y=test_data[:,-1]))

rfc = RandomForestClassifier()
rfc.fit(X=train_data[:,0:-1],y=train_data[:,-1])
print("RandomForestClassifier_Score:",rfc.score(X=test_data[:,0:-1],y=test_data[:,-1]))


