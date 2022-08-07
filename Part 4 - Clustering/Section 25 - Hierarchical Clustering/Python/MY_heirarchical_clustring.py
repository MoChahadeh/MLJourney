import numpy as np
import pandas as pd
import random

dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, 2:].values

from sklearn.preprocessing import LabelEncoder

X[:, 0] = LabelEncoder().fit_transform(X[:,0])

from sklearn.impute import SimpleImputer
X = SimpleImputer().fit_transform(X)


from sklearn.cluster import AgglomerativeClustering

clusterer = AgglomerativeClustering(compute_distances=True)
clusterer.fit(X)

distances = np.array(clusterer.distances_)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(distances.reshape(-1,1))


clusterer = AgglomerativeClustering(n_clusters = None,distance_threshold= sc.inverse_transform([[4.0]])[0,0])

Y_clust = clusterer.fit_predict(X)

print(clusterer.n_clusters_)

import matplotlib.pyplot as plt


colors = ["red", "green", "blue", "cyan", "yellow", "magenta", "purple", "gray"]

plt = plt.axes(projection = "3d")
for i in range(clusterer.n_clusters_):

    color = (random.random(), random.random(),random.random())

    plt.scatter(X[Y_clust == i, 0], X[Y_clust == i, 1],X[Y_clust == i, 2], s= 100, c= color, label=("Cluster "+ str(i)))

plt.legend()
plt.show()