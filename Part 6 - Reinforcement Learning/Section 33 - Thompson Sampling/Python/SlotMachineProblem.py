import numpy as np

rounds = 1000

numOfMachines = 6

numOfWinsPerMachine = np.zeros((numOfMachines))
numOfLossesPerMachine = np.zeros((numOfMachines))

np.random.seed(33)
convertionRates = np.random.uniform(0.01, 0.15, numOfMachines)

for i in range(numOfMachines):
    print("Convertion Rate for Machine {0}: {1: .2%}".format(i+1, convertionRates[i]))




for round in range(rounds):

    selectedMachineToPlay = -1
    maxBeta = -1

    for machine in range(numOfMachines):

        a = numOfWinsPerMachine[machine] + 1
        b = numOfLossesPerMachine[machine] + 1

        beta = np.random.beta(a, b)

        if(beta > maxBeta):
            selectedMachineToPlay = machine
            maxBeta = beta

    if(np.random.rand() <= convertionRates[selectedMachineToPlay]):
        numOfWinsPerMachine[selectedMachineToPlay] += 1
    else :
        numOfLossesPerMachine[selectedMachineToPlay] += 1


timePlayedPerMachine = numOfWinsPerMachine + numOfLossesPerMachine

for i in range(numOfMachines):

    print("Times Machine {0} was played: {1}".format(i+1, timePlayedPerMachine[i]))

print("Best Machine To Play: Machine {0}".format(np.argmax(timePlayedPerMachine) + 1))

    
        
