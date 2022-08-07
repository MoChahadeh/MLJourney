from tkinter import Label
import numpy as np
import pandas as pd

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

from sklearn.preprocessing import LabelBinarizer

Y = LabelBinarizer().fit_transform(Y)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.2)

from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, Y_train)


Y_pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_test, Y_pred)
print(accuracy)

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = classifier, X = X_train, y = Y_train, cv = 10)

print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
