import random, math

def calculateRewardRatio(priority,remainingLength):    #Kommt sowohl in Agent als auch Auctioneer vor. Refactoren in gemeinsame Oberklasse Spieler oder so. Da auch updateCores()- Methode rein
    if (priority==-1) | (remainingLength==-1):
        return -1
    if (priority==-2) | (remainingLength ==-2):
        return -1
    else:
        return priority/remainingLength
 
class HardcodedAcceptor(object):
    def __init__(self, world):
        self.amountInputChannels = (3 + (2*world.maxAmountOfOffersToOneAgent))
        self.numberOfActions = (world.maxAmountOfOffersToOneAgent + 1)
        self.rejectIndex = (self.numberOfActions - 1) #Array-Index-Logik

    def selectAction(self, observationTensor):
        observationList = observationTensor.tolist()
        assert len(observationList) == (self.amountInputChannels)
        # untige If-Abfrage bezieht sich auf ownership-Flag
        if observationList[0] == 0:
            return self.rejectIndex
        else:
            ownRewardRatio = calculateRewardRatio(observationList[1],observationList[2])
            iterator1 = iter(observationList[3:])
            OfferTuples = zip(iterator1,iterator1)
            listOfOfferdRatios = [calculateRewardRatio(priority, necessaryTime) for (priority, necessaryTime) in OfferTuples]
            if max(listOfOfferdRatios) > ownRewardRatio:
                acceptionCandidates = [(i, rewardRatio) for (i, rewardRatio) in enumerate(listOfOfferdRatios) if (rewardRatio == max(listOfOfferdRatios))]
                finalAcceptionIndex = random.sample(acceptionCandidates,1)[0][0]
                return finalAcceptionIndex
            else:
                return self.rejectIndex

class HardcodedAuctioneerAcceptor(object):
    def __init__(self, world):
        self.amountInputChannels = (3 + (2*world.maxAmountOfOffersToOneAgent))
        self.numberOfActions = (world.maxAmountOfOffersToOneAgent + 1)
        self.rejectIndex = (self.numberOfActions - 1) #Array-Index-Logik

    def selectAction(self, observationTensor):
        observationList = observationTensor.tolist()
        assert len(observationList) == (self.amountInputChannels)
        # untige If-Abfrage bezieht sich auf ownership-Flag
        if observationList[0] == 0:
            return self.rejectIndex
        else:
            ownRewardRatio = calculateRewardRatio(observationList[1],observationList[2])
            assert (ownRewardRatio == -1) # Muss so sein, weil es ja der Auktionator ist.
            iterator1 = iter(observationList[3:])
            OfferTuples = zip(iterator1,iterator1)
            listOfOfferdRatios = [calculateRewardRatio(priority, necessaryTime) for (priority, necessaryTime) in OfferTuples]
            if (max(listOfOfferdRatios) > ownRewardRatio):
                acceptionCandidates = [(i, rewardRatio) for (i, rewardRatio) in enumerate(listOfOfferdRatios) if (rewardRatio == max(listOfOfferdRatios))]
                finalAcceptionIndex = random.sample(acceptionCandidates,1)[0][0]
                return finalAcceptionIndex
            else:
                return self.rejectIndex

class HardcodedOfferer(object):
    def __init__(self,world):
        self.amountInputChannels = ((world.numberOfCores * 2) + 2)
        # Theoretisch besteht die Möglichkeit gar kein Angebot abzugeben (rejectIndex). Aber das dürfte nur in seltenen Grenzfällen relevant sein.
        self.numberOfActions = ((world.numberOfCores) + 1)
        # Im Unterschied zu oben wird hier bei Angeboten direkt die CoreID angegeben. Da die Kerne bei 1 beginnen, ist 0 die Verweigerung.
        self.rejectIndex = 0
    
    def selectAction(self, observationTensor):
        observationList = observationTensor.tolist()
        assert len(observationList) == (self.amountInputChannels)
        ownRewardRatio = calculateRewardRatio(observationList[-2],observationList[-1])

        iterator2 = iter(observationList[:-2])
        CoreStateTuples = zip(iterator2,iterator2)
        listOfCoreRatios = [calculateRewardRatio(priority, remainingLength) for (priority, remainingLength) in CoreStateTuples]
        offerCandidates = [(i, rewardRatio) for (i, rewardRatio) in enumerate(listOfCoreRatios) if (rewardRatio == min(listOfCoreRatios))]
        finalOfferCoreID = random.sample(offerCandidates,1)[0][0]   #Es muss nicht 1 hinzuaddiert werden, weil das schon in world.py geschieht

        return finalOfferCoreID