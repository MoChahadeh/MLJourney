import numpy as np
import pandas as pd

dataset = pd.read_csv('Ads_CTR_Optimisation.csv')
X = dataset.iloc[:, :].values

rounds = 10000
numOfAds = 10

numOfWinsPerAd = np.zeros((numOfAds))
numOfLossesPerAd = np.zeros((numOfAds))

for round in range(rounds):

    chosenAd = -1
    maxBeta = -1

    for ad in range(numOfAds):

        a = numOfWinsPerAd[ad] + 1
        b = numOfLossesPerAd[ad] + 1

        beta = np.random.beta(a, b)

        if(beta > maxBeta):
            chosenAd = ad
            maxBeta = beta
    
    if(X[round][chosenAd] == 1):
        numOfWinsPerAd[chosenAd] += 1
    else:
        numOfLossesPerAd[chosenAd] += 1


timesChosenPerAd = numOfWinsPerAd + numOfLossesPerAd
for i in range(numOfAds):

    print("Time Ad {0} was chosen:{1} ".format(i+1, timesChosenPerAd[i]))

print("Best Ad: {0}".format(np.argmax(timesChosenPerAd) + 1))
