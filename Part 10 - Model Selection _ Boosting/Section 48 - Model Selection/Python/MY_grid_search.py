from matplotlib.pyplot import grid
import numpy as np
import pandas as pd

dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.2)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

params = [{'C': [0.25, 0.5, 0.75, 1], 'kernel': ['linear']}, {'C': [0.25, 0.5, 0.75, 1], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
grid_search = GridSearchCV(estimator= SVC(), param_grid=params, scoring='accuracy', cv=10, n_jobs=-1)

grid_search.fit(X_train, Y_train)

best_acc = grid_search.best_score_
best_param = grid_search.best_params_

print("Accuracy: ", best_acc)
print("Best Params:", best_param)