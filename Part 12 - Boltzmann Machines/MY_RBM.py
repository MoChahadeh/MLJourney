import numpy as np
import pandas as pd

#importing training data and testing data
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

num_users = max(max(training_set[:, 0]), max(test_set[:, 0]))
num_movies = max(max(training_set[:, 1]), max(test_set[:, 1]))


def convert(data):
  new_data = []
  for id_users in range(1, num_users + 1):
    id_movies = data[:, 1] [data[:, 0] == id_users]
    id_ratings = data[:, 2] [data[:, 0] == id_users]
    ratings = np.zeros(num_movies)
    ratings[id_movies - 1] = id_ratings
    new_data.append(list(ratings))
  return new_data
training_set = np.array(convert(training_set))
test_set = np.array(convert(test_set))

# print(trainig_set)

training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1
test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

# def sigmoid(x):
#     return np.where(x >= 0, 
#                     1 / (1 + np.exp(-x)), 
#                     np.exp(x) / (1 + np.exp(x)))

class RBM():

    def __init__(self, nvis, nhid):
        self.W = np.random.randn(nhid, nvis)
        self.b = np.random.randn(1, nhid)
        self.c = np.random.randn(1, nvis)
        pass
    
    def get_hid(self, vis):
        
        hid = np.dot(vis, self.W.T) + self.b
        hid_activation = 1 / (1 + np.exp(-hid))

        new_hid = np.random.binomial(1, hid_activation)

        return hid_activation, new_hid
    
    def get_vis(self, hid):

        vis = np.dot(hid, self.W) + self.c
        vis_activation = 1 / (1 + np.exp(-vis))

        new_vis = np.random.binomial(1, vis_activation)

        return vis_activation, new_vis
    
    def train(self, vis0, visK, pHid0, pHidK):
        self.W += (np.dot(pHid0.T, vis0) - np.dot(pHidK.T, visK))*0.1
        self.b += np.sum(pHid0 - pHidK, axis=0)*0.1
        self.c += np.sum(vis0 - visK, axis=0)*0.1



nv = num_movies
nh = 5

rbm = RBM(nv, nh)

epochs = 15
batch = 10

for epochs in range(1, epochs+1) :

    train_loss = 0
    s = 0.

    for user in range(0, num_users-batch, batch):

        visK = training_set[user:user+batch]
        vis0 = training_set[user:user+batch]

        pHid0, _ = rbm.get_hid(vis0)

        for k in range(10):
            _, hidK = rbm.get_hid(visK)
            _, visK = rbm.get_vis(hidK)
            visK[vis0 == -1] = -1
        
        pHidK, _ = rbm.get_hid(visK)

        rbm.train(vis0, visK, pHid0, pHidK)
        train_loss += np.mean(np.abs(vis0[vis0 >= 0] - visK[visK >= 0]))
        s += 1.

        # print(visK[0])
        # print(vis0[0])
        # print("-----------")
    print('epoch: ' + str(epochs) + ' loss: ' + str(train_loss/s))


test_loss = 0
s = 0.
for user in range(num_users):

    rated_movies = training_set[user:user+1]
    Y_true = test_set[user:user+1]

    if len(Y_true[Y_true >= 0]) > 0:
        _, h = rbm.get_hid(rated_movies)
        _, Y_pred = rbm.get_vis(h)

        test_loss += np.mean(np.abs(Y_true[Y_true >= 0] - Y_pred[Y_true >= 0]))
        s += 1.

print( "Test Loss:  " + str(test_loss/s))
