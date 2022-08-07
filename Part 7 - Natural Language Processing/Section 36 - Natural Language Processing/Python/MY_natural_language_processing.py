from random import Random
import numpy as np
import pandas as pd

dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter="\t", quoting= 3)
X = dataset.iloc[:, 0].values
Y = dataset.iloc[:, 1].values


import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

for i in range(1000):

  X[i] = re.sub('[^a-zA-Z]', ' ', X[i])
  X[i] = X[i].lower()
  X[i] = X[i].split()
  ps = PorterStemmer()

  my_stopwords = stopwords.words("english")
  my_stopwords.remove("not")
  X[i] = [ps.stem(word) for word in X[i] if not word in set(my_stopwords)]

  X[i] = " ".join(X[i])

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=1500)

X = cv.fit_transform(X).toarray()

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB()

classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_test, Y_pred)
print(accuracy)

print(classifier.predict(cv.transform(["nice taste good"]).toarray()))
