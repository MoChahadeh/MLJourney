from tkinter import Label
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

#Reading the dataset from CSV File
dataset = pd.read_csv("Data.csv")

# Splitting to features and outcomes
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

#Imputing missing data
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer.fit(X[:, 1:])
X[:,1:]= imputer.transform(X[:, 1:])

#transforming Categorical Data
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder="passthrough")
X = np.array(ct.fit_transform(X))

le = LabelEncoder()
Y = le.fit_transform(Y)

#splitting training and testing datasets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

#Standarized Scaling
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:,3:])
X_test[:, 3: ] = sc.transform(X_test[:, 3:])

print(X_train)
print("\n")
print(X_test)
