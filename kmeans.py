##Elbow method
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.cluster import KMeans
import seaborn as sn
##take a sample csv
olympics= pd.read_csv( "olympics.csv" )
olympics.columns=map(str.upper,olympics.columns)
olympics_clean=olympics.dropna()
cluster=olympics_clean[['1', '2', '3', '4','5', '6', '7', '8','9', '10', '11', '12','13', '14', '15']]
cluster.describe()
clustervar=cluster.copy()
clustervar['1']=preprocessing.scale(clustervar['1'].astype('float64'))
clustervar['2']=preprocessing.scale(clustervar['2'].astype('float64'))
clustervar['3']=preprocessing.scale(clustervar['3'].astype('float64'))
clustervar['4']=preprocessing.scale(clustervar['4'].astype('float64'))
clustervar['5']=preprocessing.scale(clustervar['5'].astype('float64'))
clustervar['6']=preprocessing.scale(clustervar['6'].astype('float64'))
clustervar['7']=preprocessing.scale(clustervar['7'].astype('float64'))
clustervar['8']=preprocessing.scale(clustervar['8'].astype('float64'))
clustervar['9']=preprocessing.scale(clustervar['9'].astype('float64'))
clustervar['10']=preprocessing.scale(clustervar['10'].astype('float64'))
clustervar['11']=preprocessing.scale(clustervar['11'].astype('float64'))
clustervar['12']=preprocessing.scale(clustervar['12'].astype('float64'))
clustervar['13']=preprocessing.scale(clustervar['13'].astype('float64'))
clustervar['14']=preprocessing.scale(clustervar['14'].astype('float64'))
clustervar['15']=preprocessing.scale(clustervar['15'].astype('float64'))
clus_train, clus_test=train_test_split(clustervar, test_size=.3, random_state=123)
from scipy.spatial.distance import cdist
clusters=[1,2,3,4,5,6,7,8,9,10]
print(clusters)
meandist=[]
for k in clusters:
    model=KMeans(n_clusters=k)
    model.fit(clus_train)
    clusassign=model.predict(clus_train)	
    meandist.append(sum(np.mean(cdist(clus_train, model.cluster_centers_, 'euclidean'),axis=1))/clus_train.shape[0])
print(meandist)
#plt.plot(clusters,meandist)
#plt.xlabel("no of cluster")
#$plt.ylabel("avg")	
#plt.title("selecting") 

#problem showing lowest output is the optimal value of kmeans
