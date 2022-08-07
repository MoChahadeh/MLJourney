import numpy as np
import pandas as pd

dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)

from sklearn.svm import SVC

classifier = SVC(kernel="rbf")
classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(sc.transform(X_test))


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_test, Y_pred)
print(accuracy)