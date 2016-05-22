#!/usr/bin/env python
# coding=utf-8

import numpy as np
import pandas as pd

train = pd.read_csv("./train.csv",usecols=["srch_destination_id","is_booking","hotel_cluster"],nrows=10000)

srch_destination_vector = pd.read_csv("./destinations.csv")
cluster_vector = np.zeros(149)
train_list = []
for i in range(100):
    train_list.append(train[train.hotel_cluster==i])
destination_vector = pd.DataFrame()
cluster_vect = pd.DataFrame()
sum = []
count = 0
for num,train_cluster in enumerate(train_list):
    for destination_id in train_cluster["srch_destination_id"]:
        if destination_id in list(srch_destination_vector["srch_destination_id"].values):
            destination_vector = destination_vector.append(srch_destination_vector[srch_destination_vector.srch_destination_id == destination_id],ignore_index=True)
            count += 1
    sum.append(count)
    count = 0
    if num > 0:
        sum[num] += sum[num-1] 

for i,j in enumerate(sum):
    if i ==0:
        for m in range(0,j):
            cluster_vector += np.array(destination_vector.iloc[m,1:].values)
        cluster_vect[str(i)] = cluster_vector/j
        cluster_vector = np.zeros(149)
    else:
        for m in range(sum[i-1],j):
            cluster_vector += np.array(destination_vector.iloc[m,1:])
        cluster_vect[str(i)] = cluster_vector/(j-sum[i-1])
        cluster_vector = np.zeros(149)

cluster_vect.to_csv("cluster_vect.csv")

print(sum)
print(destination_vector)
print(cluster_vect)




