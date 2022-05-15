from HardcodedModules import HardcodedAcceptor, HardcodedAuctioneerAcceptor
import random, torch, numpy as np

#Empty offers are padded with -2, -2, empty cores and empty jobs are padded with -1, -1

def calculateRewardRatio(priority,remainingLength): 
    if (priority==-1)|(remainingLength==-1):
        return 0
    if (priority==-2)|(remainingLength==-2):
        return 0
    else:
        return priority/remainingLength

def gatherDividedAuctioneerObservation(world):
    auctioneerObservationTensors = []
    acceptorNetOfferIDs = []
    for i in range(world.numberOfCores):
        # (i + 1) is the core ID
        observationTensor, correspondingOfferIDs = getAuctioneerAcceptorObservationTensorAndIDs(world,(i + 1))
        auctioneerObservationTensors.append(observationTensor)
        acceptorNetOfferIDs.append(correspondingOfferIDs)

    return auctioneerObservationTensors, acceptorNetOfferIDs

def getAuctioneerAcceptorObservationTensorAndIDs(world, coreID):
        core = world.cores[coreID - 1]
        ownershipFlag = True if (core.ownerID == 0) else False
        #The net reward here is really just the current priority, since the reward is at double and the other half has been ceded to the previous owner.
        coreObservation = [{'ownershipFlag':int(ownershipFlag) , 'netReward':(core.job.priority) if ownershipFlag else (-1), 'remainingLength': (core.job.remainingLength) if ownershipFlag else (-1)}]
        offersToThatCoreToThatAgent = [{'offeredReward': offer.offeredReward,'necessaryTime': offer.necessaryTime} for offer in world.offers if (offer.recipientID == 0)&(offer.coreID==coreID)]
        #-2 is simply the default value chosen here for non-existent offers to scale the observation to a fixed length.
        #world.maxAmountOfOffersToOneAgent = (world.numberOfAgents - 1) * world.collectionLength
        offersToThatCoreToThatAgent = pad(offersToThatCoreToThatAgent, world.maxAmountOfOffersToOneAgent,{'priority': -2,'necessaryTime': -2})
        correspondingOfferIDs = [offer.offerID for offer in world.offers if (offer.recipientID == 0)&(offer.coreID==coreID)]
        correspondingOfferIDs = pad(correspondingOfferIDs,world.maxAmountOfOffersToOneAgent,-2)
        
        observationDict = coreObservation + offersToThatCoreToThatAgent
        observationTensor = torch.tensor(np.concatenate((np.array([list(coreObs.values()) for coreObs in coreObservation]).reshape(-1),np.array([list(dicti.values()) for dicti in offersToThatCoreToThatAgent]).reshape(-1))),requires_grad=False)
        
        return observationTensor, correspondingOfferIDs


class Auctioneer(object):
    def __init__(self,world):
        self.world = world
        self.auctioneerID = 0
        self.policy = [HardcodedAuctioneerAcceptor(self.world) for _ in range(self.world.numberOfCores)]
        self.ownedCores = []
    
    def updateOwnedCores(self,world):
        self.ownedCores = [core for core in world.cores if core.ownerID==0]

    def getAuctioneerState(self):
        return {'ownedCores': [core.getCoreState() for core in self.ownedCores]}
    
    def getAuctioneerAction(self, auctioneerObservationsTensor):
        
        auctioneerAction = []
        for j, acceptorNetObservation in enumerate(auctioneerObservationsTensor):
            action = self.policy[j].selectAction(acceptorNetObservation)
            auctioneerAction.append(action)

        return auctioneerAction
    
    

def pad(liste,size,padding):
    if len(liste) < size:
        return liste + [padding] * (size - len(liste))
    else:
        return liste




