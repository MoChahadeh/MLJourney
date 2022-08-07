import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

dataset = pd.read_csv("Position_Salaries.csv")

X = dataset.iloc[:, 1:-1].values
Y = dataset.iloc[:, -1].values

Yreshaped = Y.reshape((len(Y), 1))

Xscaler = StandardScaler()
Xscaled = Xscaler.fit_transform(X)

Yscaler = StandardScaler()
Yscaled = Yscaler.fit_transform(Yreshaped)

regressor = SVR()
regressor.fit(Xscaled, Yscaled)

Xsmooth = np.arange(min(X), max(X)+0.1, 0.1)
Xsmooth = Xsmooth.reshape((len(Xsmooth), 1))

plt.scatter(X, Y, color="red")
plt.plot(Xsmooth, Yscaler.inverse_transform(regressor.predict(Xscaler.transform(Xsmooth))))
plt.show()


