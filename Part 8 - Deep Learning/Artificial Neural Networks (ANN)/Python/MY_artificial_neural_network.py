from calendar import EPOCH
import numpy as np
import pandas as pd

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
Y = dataset.iloc[:, -1].values

from sklearn.preprocessing import LabelEncoder
X[:, 2] = LabelEncoder().fit_transform(X[:,2])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
X = np.array(ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder= 'passthrough').fit_transform(X))

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)

import tensorflow as tf

nn = tf.keras.models.Sequential()

nn.add(tf.keras.layers.Dense(units = 6, activation="relu"))
nn.add(tf.keras.layers.Dense(units = 6, activation="relu"))
nn.add(tf.keras.layers.Dense(units = 1, activation="sigmoid"))

nn.compile(optimizer = "adam", loss = 'binary_crossentropy', metrics = ["accuracy"])

nn.fit(X_train, Y_train, batch_size = 32, epochs = 100)

# print(nn.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)

Y_pred = nn.predict(sc.transform(X_test))

Y_pred = (Y_pred > 0.5)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_test == 1, Y_pred)
print(accuracy)
