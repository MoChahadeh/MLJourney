import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, 1:].values

from sklearn.preprocessing import LabelEncoder
X[:,0] = LabelEncoder().fit_transform(X[:,0])


from sklearn.cluster import KMeans

wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters= i)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss)
plt.show()


kmeans = KMeans(n_clusters = 5)
y_kmeans = kmeans.fit_predict(X)


newData = np.array([[0,21,80, 27]])
newpoint = kmeans.predict(newData)
print("new point Pred: ", newpoint+1)



plt = plt.axes(projection='3d')

plt.scatter(newData[:,1],newData[:, 2], newData[:,3], s= 100, c = "black", label="new point")
plt.scatter(X[y_kmeans == 0,1],X[y_kmeans == 0, 2], X[y_kmeans == 0, 3], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1,1],X[y_kmeans == 1, 2], X[y_kmeans == 1, 3], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2,1],X[y_kmeans == 2, 2], X[y_kmeans == 2, 3], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3,1],X[y_kmeans == 3, 2], X[y_kmeans == 3, 3], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4,1],X[y_kmeans == 4, 2], X[y_kmeans == 4, 3], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:,1],kmeans.cluster_centers_[:, 2], kmeans.cluster_centers_[:, 3], s = 300, c = 'yellow', label = 'Centroids')
plt.legend()
plt.show()


