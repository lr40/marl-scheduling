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
        self.rejectIndex = (self.numberOfActions - 1)

    def selectAction(self, observationTensor):
        observationList = observationTensor.tolist()
        assert len(observationList) == (self.amountInputChannels)
        # If query refers to ownership flag
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
        self.rejectIndex = (self.numberOfActions - 1)

    def selectAction(self, observationTensor):
        observationList = observationTensor.tolist()
        assert len(observationList) == (self.amountInputChannels)
        # If query refers to ownership flag
        if observationList[0] == 0:
            return self.rejectIndex
        else:
            ownRewardRatio = calculateRewardRatio(observationList[1],observationList[2])
            assert (ownRewardRatio == -1) # Must be so, because it is the auctioneer.
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
        # Theoretically, it is possible not to submit an offer at all (rejectIndex). But this should only be relevant in rare borderline cases.
        self.numberOfActions = ((world.numberOfCores) + 1)
        # In contrast to above, the CoreID is directly specified here for offers. Since the cores start at 1, 0 is the denial.
        self.rejectIndex = 0
    
    def selectAction(self, observationTensor):
        observationList = observationTensor.tolist()
        assert len(observationList) == (self.amountInputChannels)
        ownRewardRatio = calculateRewardRatio(observationList[-2],observationList[-1])

        iterator2 = iter(observationList[:-2])
        CoreStateTuples = zip(iterator2,iterator2)
        listOfCoreRatios = [calculateRewardRatio(priority, remainingLength) for (priority, remainingLength) in CoreStateTuples]
        offerCandidates = [(i, rewardRatio) for (i, rewardRatio) in enumerate(listOfCoreRatios) if (rewardRatio == min(listOfCoreRatios))]
        finalOfferCoreID = random.sample(offerCandidates,1)[0][0]   #There is no need to add 1, because this is already done in world.py

        return finalOfferCoreID