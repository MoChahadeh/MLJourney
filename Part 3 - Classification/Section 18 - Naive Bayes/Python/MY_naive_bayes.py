from telnetlib import GA
import numpy as np
import pandas as pd

dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()
classifier.fit(X_train,Y_train)

Y_pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_test, Y_pred)
print(accuracy)

import matplotlib.pylab as plt

plt.scatter(X_test[Y_pred == 0, 0], X_test[Y_pred == 0, 1], s = 100, color = "red")
plt.scatter(X_test[Y_pred == 1, 0], X_test[Y_pred == 1, 1], s = 100, color = "green")

plt.show()