import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('D:\\cognitior\\Basics of data science\\driver-data.csv')
dataset

x = dataset.iloc[:, [1,2]].values
x

from sklearn.cluster import KMeans

wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++')
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11), wcss)

kmeans = KMeans(n_clusters=2, init='k-means++')
y_kmeans = kmeans.fit_predict(x)
y_kmeans

dataset

pd.concat([dataset, pd.DataFrame(y_kmeans)], axis=1)

plt.scatter(x[y_kmeans==0,0], x[y_kmeans==0,1], s=100, c='red', label='cluster1')
plt.scatter(x[y_kmeans==1,0], x[y_kmeans==1,1], s=100, c='pink', label='cluster1')

import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x,method='ward'))
plt.title('Dendrogram')

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', 
                             linkage='ward')

y_hc = hc.fit_predict(x)
y_hc

plt.scatter(x[y_hc==0,0], x[y_hc==0,1], s=100, c='red', label='cluster1')
plt.scatter(x[y_hc==1,0], x[y_hc==1,1], s=100, c='pink', label='cluster1')

pd.concat([dataset, pd.DataFrame(y_hc)], axis=1)