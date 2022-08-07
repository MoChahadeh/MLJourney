import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

dataset = pd.read_csv("Position_Salaries.csv")

X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

linearRegressor = LinearRegression()
linearRegressor.fit(X, Y)


polyProcessor = PolynomialFeatures(degree = 6)
XPoly = polyProcessor.fit_transform(X)

polyRegressor = LinearRegression()
polyRegressor.fit(XPoly, Y)


plt.scatter(X, Y, color = "red")
plt.plot(X, linearRegressor.predict(X))
plt.title("Linear Regression model")
plt.show()

plt.scatter(X, Y, color = "red")
plt.plot(X, polyRegressor.predict(XPoly))
plt.title("Polynomial Regression model")
plt.show()

print("Linear Prediction:", linearRegressor.predict([[6.5]]))
print("Non-linear Prediction (6th degree):", polyRegressor.predict(polyProcessor.fit_transform([[6.5]])))