import numpy as np
import tensorflow as tf

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")

# a = np.array(y_train)
# y_train= np.zeros((a.size, a.max()+1))
# y_train[np.arange(a.size),a] = 1

# a = np.array(y_test)
# y_test= np.zeros((a.size, a.max()+1))
# y_test[np.arange(a.size),a] = 1

new_X_train = np.ndarray((60000,784))

for i in range(np.size(X_train,0)):
    new_X_train[i] = np.concatenate(X_train[i])

new_X_test = np.ndarray((np.size(X_test,0), 784))

for i in range(np.size(X_test,0)):
    new_X_test[i] = np.concatenate(X_test[i])


nn = tf.keras.models.Sequential()

nn.add(tf.keras.layers.Dense(units = 32, activation="relu"))
nn.add(tf.keras.layers.Dense(units = 32, activation="relu"))
nn.add(tf.keras.layers.Dense(units = 16, activation="relu"))
nn.add(tf.keras.layers.Dense(units = 10, activation="softmax"))

nn.compile(optimizer = "adam", loss = 'sparse_categorical_crossentropy', metrics = ["accuracy"])

nn.fit(new_X_train, y_train, batch_size = 250, epochs = 20)


Y_pred = nn.predict(new_X_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, np.argmax(Y_pred, 1))
print("Test Accuracy: " + str(accuracy))