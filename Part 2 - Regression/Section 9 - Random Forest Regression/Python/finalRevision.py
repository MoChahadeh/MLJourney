import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits import mplot3d
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

dataset = pd.read_csv("50_Startups.csv")

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')

X = ct.fit_transform(X)
X = X[:, 1:-1]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state= 0)

regressor = RandomForestRegressor(n_estimators= 11, random_state=0)
regressor.fit(X_train, Y_train)

Y_pred = regressor.predict(X_test)

errorRate = np.divide(abs(np.subtract(Y_test,Y_pred)), Y_test)
meanError = np.mean(errorRate)

print(errorRate)
print("Mean Error: ",meanError)

fig = plt.figure()
ax = plt.axes(projection='3d')

xIndex = 3
yIndex = 2
ax.scatter(X_test[: ,xIndex], X_test[:, yIndex], Y_test, color = "red")
ax.scatter(X_test[: ,xIndex], X_test[:, yIndex], Y_pred, color = "blue")
plt.show()
