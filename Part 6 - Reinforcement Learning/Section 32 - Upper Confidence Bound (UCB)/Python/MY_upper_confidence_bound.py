import numpy as np
import pandas as pd
import math

dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

totalRounds = 10000
numOfAds = 10

adSelectedEachRound = []

numOfSelectionsForAd = [0] * numOfAds
sumOfRewardsForAd = [0] * numOfAds
totalRewards = 0

for round in range(0, totalRounds):

    ad = 0
    maxUpperConfidenceBound = 0
    for i in range(0, numOfAds):
        if(numOfSelectionsForAd[i] > 0):
            averageReward = sumOfRewardsForAd[i] / numOfSelectionsForAd[i]
            deltaI = math.sqrt((3/2) * math.log(round+1)/numOfSelectionsForAd[i])
            upperConfidenceBound = averageReward + deltaI
        else:
            upperConfidenceBound = 1e400
        if upperConfidenceBound > maxUpperConfidenceBound:
            maxUpperConfidenceBound = upperConfidenceBound
            ad = i
    
    adSelectedEachRound.append(ad)
    numOfSelectionsForAd[ad] += 1
    reward = dataset.values[round, ad]
    sumOfRewardsForAd[ad] += reward
    totalRewards += reward

import matplotlib.pyplot as plt
plt.hist(adSelectedEachRound)
plt.show()