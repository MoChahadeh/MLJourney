from distutils.log import error
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from mpl_toolkits import mplot3d

dataset = pd.read_csv("50_Startups.csv")

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values


ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder="passthrough")
X = ct.fit_transform(X)
X = X[:, 1:]

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2 , random_state=0)

regressor = DecisionTreeRegressor()
regressor.fit(Xtrain, Ytrain)

Ypredicted = regressor.predict(Xtest)

errorRates = np.divide(abs(np.subtract(Ypredicted, Ytest)),Ytest)

meanError = np.mean(errorRates)

print(errorRates)
print("Mean Error:", meanError)

plt.scatter(Xtest[:, 3], Ytest, color="red")
plt.scatter(Xtest[:, 3], Ypredicted, color = "blue")
plt.show()


# ax = plt.axes(projection="3d")


# ax.scatter3D(Xtest[:, 3], Xtest[:, 4], Ytest, color = "red")
# ax.scatter3D(Xtest[:, 3], Xtest[:, 4], Ypredicted, color = "blue")
# ax.show()


# fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (32,32), dpi=250)
# _ = plot_tree(regressor, filled=True, fontsize=8, )
# fig.savefig('tree.png')


# ax = plt.axes(projection='3d')

# # Data for a three-dimensional line
# zline = np.linspace(0, 15, 1000)
# xline = np.sin(zline)
# yline = np.cos(zline)
# ax.plot3D(xline, yline, zline, 'gray')

# # Data for three-dimensional scattered points
# zdata = 15 * np.random.random(100)
# xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
# ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
# ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')
