import numpy as np
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.decomposition import PCA

component=7
destination_vector = pd.read_csv("./destinations.csv")

pca = PCA(n_components=component)
dest_matrix = np.array(destination_vector.iloc[:,1:])
dest_matrix = pca.fit_transform(X=dest_matrix)

dest_matrix_df = pd.DataFrame(dest_matrix,columns=list(range(component)),index=destination_vector["srch_destination_id"])

rows= 10000000
train_a = pd.read_csv("./train.csv",usecols=["srch_rm_cnt","srch_destination_type_id","srch_destination_id","user_location_country","user_location_region","user_location_city","hotel_continent","hotel_country","hotel_market","hotel_cluster"],nrows=rows)
#most_dest_id = train_a.groupby("srch_destination_id").apply(pd.DataFrame.count)["srch_destination_id"].apply(pd.DataFrame.max())


id_covered = set(destination_vector["srch_destination_id"])

type_id = train_a.loc[:,["srch_destination_id","srch_destination_type_id"]]
most_id = type_id.groupby("srch_destination_type_id").apply(pd.DataFrame.mode)

most_id = most_id.sort(columns="srch_destination_type_id")
most_id = most_id.set_index("srch_destination_type_id")

#most_id = np.array(most_id.values)
#print(most_id)
train_b = np.zeros((train_a.shape[0],train_a.shape[1]+component))
train_b[:,:train_a.shape[1]] = np.array(train_a)
for num,id in enumerate(list(train_b[:,-6-component])):    
    if id not in id_covered:
        type_id = train_b[num,-5-component]
        new_id = most_id.loc[int(type_id)].values[0]
        if new_id not in id_covered:
            new_id = 8250
        train_b[num,-6-component] = new_id       
    train_b[num,-component:] = dest_matrix_df.loc[int(train_b[num,-6-component])].values


hotel_cluster = np.array(train_b[:,-component-1])   #为什么一定要加个np.array??因为hotel_cluster不是预先定义的array，如果直接传进去会导致传个对象。
train_b[:,-component-1] = train_b[:,-1]
train_b[:,-1] = hotel_cluster[:]
print(hotel_cluster[:10])
print(train_b[:10,-1])

train_b[:,0] = train_b[:,0]*10
train_b[:,2] = train_b[:,2]/100
train_b[:,3] = train_b[:,3]*100
train_b[:,4] = train_b[:,4]/10
train_b[:,5:7] = train_b[:,5:7]*100
train_b[:,7] = train_b[:,7]*10
train_b[:,8:-1] = train_b[:,8:-1]*100
#do or not diff 1%
train_b[:,:-1] = MinMaxScaler(feature_range=(-1500,1500)).fit_transform(X=train_b[:,:-1])
train_data = train_b[0:int(-rows/10),:]
test_data = train_b[int(-rows/10):,:]


dtc = DecisionTreeClassifier()
dtc.fit(X=train_data[:,0:-1],y=train_data[:,-1])
#joblib.dump(dtc,"dtc.model")
print("dtc score:",dtc.score(X=test_data[:,0:-1],y=test_data[:,-1]))

'''
svc = SVC()
svc.fit(X=train_data[:,0:-1],y=train_data[:,-1])
print("svc score:",svc.score(X=test_data[:,0:-1],y=test_data[:,-1]))

'''

