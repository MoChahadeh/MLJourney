import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression


dataset = pd.read_csv("50_Startups.csv")

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values


ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [3])], remainder="passthrough")

X = np.array(ct.fit_transform(X))

X_optimal = X[:, [3, 5]]


X_train, X_test, Y_train, Y_test = train_test_split(
    X_optimal, Y, test_size=0.2, random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
Y_predicted = regressor.predict(X_test)


featureIndex = 1
plt.scatter(X_test[:,featureIndex], Y_test, color="red")
plt.plot(np.sort(X_test[:,featureIndex]), regressor.predict(X_test[X_test[:,featureIndex].argsort()]))
plt.ylabel("Profit")
plt.show()

# Calculating Error:
def getError():
    global errorPercentage, Y_test, Y_predicted, meanError
    print("\n")
    errorPercentage = np.divide(abs(np.subtract(Y_test, Y_predicted)), Y_test)
    print(errorPercentage)

    meanError = np.mean(errorPercentage)
    print("MEAN ERROR:", meanError)
    print("\n")

getError()


def getFullFormula():

    b0 = regressor.predict([[0,0]])[0]
    print("Intercept:", b0)

    b1 = regressor.predict([[1, 0]])[0] - b0
    print("b1:", b1)

    b2 = regressor.predict([[0, 1]])[0] - b0
    print("b2:", b2)

getFullFormula()


# # Manually doing BackElimination:
# X = X[:, 1:]  # Removing the dummy variable

# X = np.append(np.ones((50, 1)).astype(int), X.astype(np.float64), axis=1)

# X_optimal = X[:, [0, 1, 2, 3, 4, 5]]
# regressor_OLS = sm.OLS(endog=Y, exog=X_optimal).fit()

# X_train, X_test, Y_train, Y_test = train_test_split(
#     X_optimal, Y, test_size=0.2, random_state=0)
# regressor = LinearRegression()
# regressor.fit(X_train, Y_train)
# Y_predicted = regressor.predict(X_test)
# getError()

# X_optimal = X[:, [0, 1, 3, 4, 5]]
# regressor_OLS = sm.OLS(endog=Y, exog=X_optimal).fit()

# X_train, X_test, Y_train, Y_test = train_test_split(
#     X_optimal, Y, test_size=0.2, random_state=0)
# regressor = LinearRegression()
# regressor.fit(X_train, Y_train)
# Y_predicted = regressor.predict(X_test)
# getError()

# X_optimal = X[:, [0, 3, 4, 5]]
# regressor_OLS = sm.OLS(endog=Y, exog=X_optimal).fit()

# X_train, X_test, Y_train, Y_test = train_test_split(
#     X_optimal, Y, test_size=0.2, random_state=0)
# regressor = LinearRegression()
# regressor.fit(X_train, Y_train)
# Y_predicted = regressor.predict(X_test)
# getError()

# regressor_OLS = sm.OLS(endog=Y, exog=X_optimal).fit()

# X_train, X_test, Y_train, Y_test = train_test_split(
#     X_optimal, Y, test_size=0.2, random_state=0)
# regressor = LinearRegression()
# regressor.fit(X_train, Y_train)
# Y_predicted = regressor.predict(X_test)
# getError()

# X_optimal = X[:, [0, 3]]
# regressor_OLS = sm.OLS(endog=Y, exog=X_optimal).fit()

# X_train, X_test, Y_train, Y_test = train_test_split(
#     X_optimal, Y, test_size=0.2, random_state=0)
# regressor = LinearRegression()
# regressor.fit(X_train, Y_train)
# Y_predicted = regressor.predict(X_test)
# getError()
