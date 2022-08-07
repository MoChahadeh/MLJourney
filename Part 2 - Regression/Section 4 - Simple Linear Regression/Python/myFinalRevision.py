import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("Salary_Data.csv")

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


regressor = LinearRegression()

regressor.fit(X_train, Y_train)

Y_predicted = regressor.predict(X_test)

plt.scatter(X_train, Y_train, color="red")
plt.plot(X_train, regressor.predict(X_train))
plt.title("Experience vs Salary (Training Data)")
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.show()

plt.scatter(X_test, Y_test, color="red")
plt.plot(X_test, Y_predicted)
plt.title("Experience vs Salary (Test Data)")
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.show()
