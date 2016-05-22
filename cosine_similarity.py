#!/usr/bin/env python
# coding=utf-8

import pandas as pd
import numpy as np

cluster_vector = pd.read_csv("./cluster_vect.csv")
train = pd.read_csv("./train.csv",usecols=["srch_destination_id","is_booking","hotel_cluster"],nrows=30)
train = train[train.is_booking==1]

destination_vector = pd.read_csv("./destinations.csv")

def cos_similarity(a,b):
    similarity = a.dot(b.T)/np.sqrt(np.square(a).sum())/np.sqrt(np.square(b).sum())
    return similarity

for vector in train["srch_destination_id"].values:
    dest_vector = destination_vector[destination_vector.srch_destination_id == vector].values[0][1:]
    matrix = np.array(cluster_vector.values)
    for i in range(100):
        similarity = cos_similarity(dest_vector,matrix[:,i+1])
        print(i,similarity)
    print(vector)

print(train["hotel_cluster"].values)
