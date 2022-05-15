from random import Random
from DQNmodules import *
from PPOmodules import *
from HardcodedModules import *
from env.world import *
import env.world as welt
import torch, torch.optim as optim
import copy

from collections import namedtuple

def pad(liste,size,padding):
    if len(liste) < size:
        return liste + [padding] * (size - len(liste))
    else:
        return liste[:size]

class Agent(object):
    IDCounter = 1
    def __init__(self,world):
        self.world = world
        self.agentID = Agent.IDCounter
        Agent.IDCounter += 1
        self.collection = welt.JobCollection(world.collectionLength)
        self.ownedCores = []
    
    def updateOwnedCores(self,world):
        self.ownedCores = [core for core in world.cores if core.ownerID==self.agentID]
    
    def resetWaitAttributeForTheEntireCollection(self):
        for job in self.collection:
            job.wait = False
     
    def getAgentState(self):
        resultDictionary = {'agentID': self.agentID, 'ownedCores': [core.getCoreState() for core in self.ownedCores], 'collectionState': self.collection.getCollectionState()}
        return resultDictionary
    
    def fillCollectionRandomly(self,amountOfNewEntries):
            for _ in range(amountOfNewEntries):
                random1 = random.random()
                for i in range(len(self.world.accProbabilities)):
                    if (random1 < self.world.accProbabilities[i]):
                        randomIndex = i
                        break
                chosenLength = self.world.possibleJobLengths[randomIndex]
                chosenPriority = self.world.possibleJobPriorities[randomIndex]                                #randomIndex also specifies the job type
                self.collection.insertJob(welt.Job(self.agentID,chosenPriority,chosenLength,False,self.world.round,randomIndex))

class AggregatedAgent(Agent):
    def __init__(self,world):
        super().__init__(world)
    
    def gatherObservations(self):
        acceptorSideObservationTensor, offerIDs = self.getAcceptorSideObservationTensorAndIDs()
        offerSideObservationTensor = self.getOfferSideObservationTensor()
        return acceptorSideObservationTensor, offerIDs, offerSideObservationTensor
    
    def getAcceptorSideObservationTensorAndIDs(self):
        observationAccumulator = torch.tensor([])
        offerIDsAccumulator = []
        for core in self.world.cores:
            ownershipFlag = True if (core.ownerID == self.agentID) else False
            # The net reward here is really just the current priority, since the reward is at double and the other half has been ceded to the previous owner.
            coreObservation = [{'ownershipFlag':int(ownershipFlag) , 'bruttoReward':(core.job.priority) if ownershipFlag else (-1), 'remainingLength': (core.job.remainingLength) if ownershipFlag else (-1)}]
            offersToThatCoreToThatAgent = [{'offerdReward': offer.offeredReward,'necessaryTime': offer.necessaryTime} for offer in self.world.offers if (offer.recipientID == self.agentID)&(offer.coreID==core.coreID)]
            #-2 is simply the default value chosen here for non-existent offers to scale the observation to a fixed length.
            #world.maxAmountOfOffersToOneAgent = (world.numberOfAgents - 1) * world.collectionLength
            offersToThatCoreToThatAgent = pad(offersToThatCoreToThatAgent, self.world.maxAmountOfOffersToOneAgent,{'priority': -2,'necessaryTime': -2})
            correspondingOfferIDs = [offer.offerID for offer in self.world.offers if (offer.recipientID == self.agentID)&(offer.coreID==core.coreID)]
            correspondingOfferIDs = pad(correspondingOfferIDs,self.world.maxAmountOfOffersToOneAgent,-2)
            #These offerIDs will become important later, since the acceptance network selects an offer for acceptance by index action.
            observationTensor = torch.tensor(np.concatenate((np.array([list(coreObs.values()) for coreObs in coreObservation]).reshape(-1),np.array([list(dicti.values()) for dicti in offersToThatCoreToThatAgent]).reshape(-1))),requires_grad=False)
            observationAccumulator = torch.cat((observationAccumulator,observationTensor),0)
            offerIDsAccumulator.append(correspondingOfferIDs)
        return observationAccumulator, offerIDsAccumulator
    
    def getOfferSideObservationTensor(self):
        coresObservationList = [(core.job.priority,core.job.remainingLength) for core in self.world.cores]
        queueObservationList = [(job.priority,job.remainingLength) for job in self.collection]
        observation = torch.tensor(coresObservationList+queueObservationList).flatten()
        
        return observation

class DividedAgent(Agent):
    def __init__(self,world):
        super().__init__(world)
        self.maxVisibleAcceptedOffers = world.maxVisibleOffers
    
    def gatherObservations(self):
        acceptorObservationTensors = []
        acceptorNetOfferIDs = []
        for i in range(self.world.numberOfCores):
            # (i + 1) is the core ID
            observationTensor, correspondingOfferIDs = self.getAcceptorObservationTensorAndIDs(i + 1)
            acceptorObservationTensors.append(observationTensor)
            acceptorNetOfferIDs.append(correspondingOfferIDs)
        
        offerObservationTensors = []
        for i in range(self.world.collectionLength):
            # i is also the queuePostion managed by this offerNet.
            observationTensor = self.getOfferNetObservationTensor(i)
            offerObservationTensors.append(observationTensor)
        
        return acceptorObservationTensors,acceptorNetOfferIDs,offerObservationTensors

    def getAcceptorObservationTensorAndIDs(self, coreID):
        core = self.world.cores[coreID - 1]
        ownershipFlag = True if (core.ownerID == self.agentID) else False
        # The net reward here is really just the current priority, since the reward is at double and the other half has been ceded to the previous owner.
        coreObservation = [{'ownershipFlag':int(ownershipFlag),'bruttoReward':(core.job.priority) if ownershipFlag else (-1), 'remainingLength': (core.job.remainingLength) if ownershipFlag else (-1)}]
        offersToThatCoreToThatAgent = [{'offerdReward': offer.offeredReward,'necessaryTime': offer.necessaryTime} for offer in self.world.offers if (offer.recipientID == self.agentID)&(offer.coreID==coreID)]
        #-2 is simply the default value chosen here for non-existent offers to scale the observation to a fixed length.
        #world.maxAmountOfOffersToOneAgent = (world.numberOfAgents - 1) * world.collectionLength
        offersToThatCoreToThatAgent = pad(offersToThatCoreToThatAgent, self.world.maxAmountOfOffersToOneAgent,{'priority': -2,'necessaryTime': -2})
        correspondingOfferIDs = [offer.offerID for offer in self.world.offers if (offer.recipientID == self.agentID)&(offer.coreID==coreID)]
        correspondingOfferIDs = pad(correspondingOfferIDs,self.world.maxAmountOfOffersToOneAgent,-2)
        #These offerIDs will become important later, since the acceptance network selects an offer for acceptance by index action.
        observationDict = coreObservation + offersToThatCoreToThatAgent
        observationTensor = torch.tensor(np.concatenate((np.array([list(coreObs.values()) for coreObs in coreObservation]).reshape(-1),np.array([list(dicti.values()) for dicti in offersToThatCoreToThatAgent]).reshape(-1))),requires_grad=False)
        
        return observationTensor, correspondingOfferIDs
    
    def getAcceptorObservationTensorAndIDs1(self, coreID):
        core = self.world.cores[coreID - 1]
        ownershipFlag = True if (core.ownerID == self.agentID) else False

        coreObservation = [{'ownershipFlag':int(ownershipFlag),'bruttoReward':(core.job.priority) if ownershipFlag else (-1), 'remainingLength': (core.job.remainingLength) if ownershipFlag else (-1)}]
        offersToThatCoreToThatAgent = [{'offerdReward': offer.offeredReward,'necessaryTime': offer.necessaryTime} for offer in self.world.offers if (offer.recipientID == self.agentID)&(offer.coreID==coreID)]

        #world.maxAmountOfOffersToOneAgent = (world.numberOfAgents - 1) * world.collectionLength
        offersToThatCoreToThatAgent = pad(offersToThatCoreToThatAgent, self.world.maxAmountOfOffersToOneAgent,{'priority': -2,'necessaryTime': -2})
        correspondingOfferIDs = [offer.offerID for offer in self.world.offers if (offer.recipientID == self.agentID)&(offer.coreID==coreID)]
        correspondingOfferIDs = pad(correspondingOfferIDs,self.world.maxAmountOfOffersToOneAgent,-2)
        
        
        acceptedOfferRatios = [{'priority': entry[0], 'remainingLength': entry[1]} for entry in self.world.acceptedOfferRatios[self.agentID - 1][coreID - 1]]
        acceptedOfferRatios = pad(acceptedOfferRatios,self.maxVisibleAcceptedOffers,{'priority': -3,'necessaryTime': -3})
        

        observationTensor = torch.tensor(np.concatenate((np.array([list(coreObs.values()) for coreObs in coreObservation]).reshape(-1),\
            np.array([list(dicti.values()) for dicti in offersToThatCoreToThatAgent]).reshape(-1),\
            np.array([list(dicti.values()) for dicti in acceptedOfferRatios]).reshape(-1))),requires_grad=False)
        
        return observationTensor, correspondingOfferIDs
    
    def getOfferNetObservationTensor(self, queuePosition):
        coresObservationList = [{'priority': core.job.priority,'remainingLength': core.job.remainingLength} for core in self.world.cores]

        queuePositionJob = self.collection[queuePosition]
        queuePositionObservationList = [{'priority': queuePositionJob.priority, 'remainingLength': queuePositionJob.remainingLength}]

        observationDict = coresObservationList + queuePositionObservationList
        observationTensor = torch.tensor(np.concatenate((np.array([list(coresObs.values()) for coresObs in coresObservationList]).reshape(-1) , np.array([list(queueObs.values()) for queueObs in queuePositionObservationList]).reshape(-1))),requires_grad=False)

        return observationTensor
    

class DividedFixPriceDQNAgent(DividedAgent):
    def __init__(self,world,env):
        super().__init__(world)
        self.policy = {'AcceptorNets': [DQNAcceptorNet(world,env) for core in world.cores], 'OfferNets': [DQNOfferNet(world,env) for slot in range(world.collectionLength)]}
        self.target = copy.deepcopy(self.policy)
        self.initializeTargetNets()
        self.optimizers = { 'acceptorOptimizers': [optim.Adam(acceptorNet.parameters()) for acceptorNet in self.policy['AcceptorNets']] ,
                            'offerOptimizers': [optim.Adam(offerNet.parameters()) for offerNet in self.policy['OfferNets']]}
        self.memories = {   'acceptorMemories': [ReplayMemory(env.REPLAY_MEMORY_SIZE) for _ in self.policy['AcceptorNets']],
                            'offerMemories': [ReplayMemory(env.REPLAY_MEMORY_SIZE) for _ in self.policy['OfferNets']]}

    def updateTargetNets(self):
        for i, acceptorPolicyNet in enumerate(self.policy['AcceptorNets']):
            self.target['AcceptorNets'][i].load_state_dict(acceptorPolicyNet.state_dict())
                  
        for i, offerNet in enumerate(self.policy['OfferNets']):
            self.target['OfferNets'][i].load_state_dict(offerNet.state_dict())
            
    def initializeTargetNets(self):
        for i, acceptorPolicyNet in enumerate(self.policy['AcceptorNets']):
            self.target['AcceptorNets'][i].load_state_dict(acceptorPolicyNet.state_dict())
            self.target['AcceptorNets'][i].eval()
        
        for i, offerNet in enumerate(self.policy['OfferNets']):
            self.target['OfferNets'][i].load_state_dict(offerNet.state_dict())
            self.target['OfferNets'][i].eval()

    
    def getActions(self,offerNetObservationTensors,acceptorNetObservationTensors):
        offerNetActions = []
        for i, offerNetObservation in enumerate(offerNetObservationTensors):
            action = self.policy['OfferNets'][i].selectAction(offerNetObservation)
            offerNetActions.append(action)
        
        acceptorNetActions = []
        for j, acceptorNetObservation in enumerate(acceptorNetObservationTensors):
            action = self.policy['AcceptorNets'][j].selectAction(acceptorNetObservation)
            acceptorNetActions.append(action)
        
        return offerNetActions, acceptorNetActions

class AggregatedFixPricePPOAgent(AggregatedAgent):
    def __init__(self,world,env):
        super().__init__(world)
        self.env = env
        self.policy = {'AcceptorUnit': AggregatedAcceptorPPO(world,env),'OfferUnit': AggregatedOfferPPO(world,env)}

    def getActions(self,offerObservations,acceptorObservations):
        aggregatedAcceptorAction = self.policy['AcceptorUnit'].selectAction(acceptorObservations)
        aggregatedOfferAction = self.policy['OfferUnit'].selectAction(offerObservations)

        #The function transforms the number representing an aggregated action back into the individual actions that can be processed by the system.
        nd_aggregatedAcceptorAction = numberToNDimensionalAction(aggregatedAcceptorAction,(self.world.maxAmountOfOffersToOneAgent + 1),self.world.numberOfCores)
        nd_aggregatedOfferAction = numberToNDimensionalAction(aggregatedOfferAction,(self.world.numberOfCores + 1),self.world.collectionLength)

        return nd_aggregatedOfferAction, nd_aggregatedAcceptorAction
    
    def updateParts(self):
        self.policy['AcceptorUnit'].update()
        self.policy['OfferUnit'].update()
    
    def saveRewards(self, offerUnitReward, agentReward):
        self.policy['AcceptorUnit'].buffer.rewards.append(agentReward)
        self.policy['OfferUnit'].buffer.rewards.append(offerUnitReward[0])

class FullyAggregatedFixPricePPOAgent(Agent):
    def __init__(self,world,env):
        super().__init__(world)
        self.env = env
        self.policy = {'FullyAggregatedUnit':FullyAggregatedPPO(world,env)}
    
    def gatherObservations(self):
        ### Acceptor-Side-Observation Begin ###
        acceptorObservationAccumulator = torch.tensor([])
        offerIDsAccumulator = []
        for core in self.world.cores:
            ownershipFlag = True if (core.ownerID == self.agentID) else False

            coreObservation = [{'ownershipFlag':int(ownershipFlag) , 'bruttoReward':(core.job.priority) if ownershipFlag else (-1), 'remainingLength': (core.job.remainingLength) if ownershipFlag else (-1)}]
            offersToThatCoreToThatAgent = [{'offerdReward': offer.offeredReward,'necessaryTime': offer.necessaryTime} for offer in self.world.offers if (offer.recipientID == self.agentID)&(offer.coreID==core.coreID)]
            
            #world.maxAmountOfOffersToOneAgent = (world.numberOfAgents - 1) * world.collectionLength
            offersToThatCoreToThatAgent = pad(offersToThatCoreToThatAgent, self.world.maxAmountOfOffersToOneAgent,{'priority': -2,'necessaryTime': -2})
            correspondingOfferIDs = [offer.offerID for offer in self.world.offers if (offer.recipientID == self.agentID)&(offer.coreID==core.coreID)]
            correspondingOfferIDs = pad(correspondingOfferIDs,self.world.maxAmountOfOffersToOneAgent,-2)

            observationTensor = torch.tensor(np.concatenate((np.array([list(coreObs.values()) for coreObs in coreObservation]).reshape(-1),np.array([list(dicti.values()) for dicti in offersToThatCoreToThatAgent]).reshape(-1))),requires_grad=False)
            acceptorObservationAccumulator = torch.cat((acceptorObservationAccumulator,observationTensor),0)
            offerIDsAccumulator.append(correspondingOfferIDs)
        ### Acceptor-Side-Observation End ###
        acceptorObservationAccumulator, offerIDsAccumulator
        ### Offer-Side-Observation Begin ###
        coresObservationList = [(core.job.priority,core.job.remainingLength) for core in self.world.cores]
        queueObservationList = [(job.priority,job.remainingLength) for job in self.collection]
        offerObservation = torch.tensor(coresObservationList+queueObservationList).flatten()
        ### Offer-Side-Observation End ###
        
        return acceptorObservationAccumulator, offerIDsAccumulator, offerObservation
    
    def getActions(self,offerObservations,acceptorObservations):
        fullyAggregatedObservation = torch.cat((offerObservations,acceptorObservations))
        fullyAggregatedAction = self.policy['FullyAggregatedUnit'].selectAction(fullyAggregatedObservation)
        #The fully-aggregated plot lies in a rectangle whose sides denote the acceptor plot and the offer plot#
        divisor = (self.world.numberOfCores + 1)**self.world.collectionLength
        f_A_acceptorAction = fullyAggregatedAction // divisor   #By modulo(%) and residueless division(//) the row and column of the plot in the rectangle are determined.
        f_A_offerAction = fullyAggregatedAction % divisor

        #The function transforms the number representing an aggregated action back into the individual actions that can be processed by the system.
        nd_aggregatedAcceptorAction = numberToNDimensionalAction(f_A_acceptorAction,(self.world.maxAmountOfOffersToOneAgent + 1),self.world.numberOfCores)
        nd_aggregatedOfferAction = numberToNDimensionalAction(f_A_offerAction,(self.world.numberOfCores + 1),self.world.collectionLength)

        return nd_aggregatedOfferAction, nd_aggregatedAcceptorAction
    
    def updateParts(self):
        self.policy['FullyAggregatedUnit'].update()
    
    def saveRewards(self, fullyAggregatedReward):
        #works with the semi-aggregated rewards
        self.policy['FullyAggregatedUnit'].buffer.rewards.append(fullyAggregatedReward)

class DividedFixedPricePPOAgent(DividedAgent):
    def __init__(self,world,env):
        super().__init__(world)
        self.env = env
        self.policy = {'AcceptorNets': [AcceptorPPO(world,env) for core in world.cores], 'OfferNets': [OfferPPO(world,env) for slot in range(world.collectionLength)]}
        
    
    def getActions(self,offerNetObservationTensors,acceptorNetObservationTensors):
        offerNetActions = []
        for i, offerNetObservation in enumerate(offerNetObservationTensors):
            action = self.policy['OfferNets'][i].selectAction(offerNetObservation)
            offerNetActions.append(action)
        
        acceptorNetActions = []
        for j, acceptorNetObservation in enumerate(acceptorNetObservationTensors):
            action = self.policy['AcceptorNets'][j].selectAction(acceptorNetObservation)
            acceptorNetActions.append(action)
        
        return offerNetActions, acceptorNetActions
    
    def clearBuffers(self):
        for acceptorNet in self.policy['AcceptorNets']:
            acceptorNet.buffer.clear()
        
        for offerNet in self.policy['OfferNets']:
            offerNet.buffer.clear()
    
    def updateParts(self):
        for acceptorNet in self.policy['AcceptorNets']:
            acceptorNet.update()
        
        for offerNet in self.policy['OfferNets']:
            offerNet.update()
    
    def saveRewards(self, offerNetRewards, acceptorNetRewards):
        for i, reward in enumerate(offerNetRewards):
            self.policy['OfferNets'][i].buffer.rewards.append(reward[0])
        
        for i, reward in enumerate(acceptorNetRewards):
            self.policy['AcceptorNets'][i].buffer.rewards.append(reward[0])

class DividedFixedPriceGloballySharedPPOAgent(DividedAgent):
    def __init__(self,world,env):
        super().__init__(world)
        self.world = world
        self.env = env
        self.policy = {'AcceptorNet': env.sharedAcceptorNet, 'OfferNet': env.sharedOfferNet}
        self.CENTRALISATION_SAMPLE = self.env.CENTRALISATION_SAMPLE
    
    def getActions(self,offerNetObservationTensors,acceptorNetObservationTensors):
        offerNetActions = []
        for i, offerNetObservation in enumerate(offerNetObservationTensors):
            action = self.policy['OfferNet'].selectAction(self.agentID,i,offerNetObservation,)
            offerNetActions.append(action)
        
        acceptorNetActions = []
        for j, acceptorNetObservation in enumerate(acceptorNetObservationTensors):
            action = self.policy['AcceptorNet'].selectAction(self.agentID,j,acceptorNetObservation,)
            acceptorNetActions.append(action)
        
        return offerNetActions, acceptorNetActions
    
    def saveRewards(self, offerNetRewards, acceptorNetRewards):
        for i, reward in enumerate(offerNetRewards):
            self.policy['OfferNet'].buffers[self.agentID - 1][i].rewards.append(reward[0])
        
        for i, reward in enumerate(acceptorNetRewards):
            self.policy['AcceptorNet'].buffers[self.agentID - 1][i].rewards.append(reward[0])
    
    def updateParts(self):
        ...

class DividedFreePricePPOAgent(DividedAgent):
    def __init__(self,world,env):
        super().__init__(world)
        self.world = world
        self.env = env
        self.policy = {'AcceptorNets': [AcceptorPPO(world,env) for core in world.cores], 'OfferNets': [FreePriceOfferPPO(world,env) for slot in range(world.collectionLength)]}
    
    def getActions(self,offerNetObservationTensors,acceptorNetObservationTensors):
        offerNetActions = []
        for i, offerNetObservation in enumerate(offerNetObservationTensors):
            action = self.policy['OfferNets'][i].selectAction(offerNetObservation)
            # Tuples are appended here.
            offerNetActions.append(action)
        
        acceptorNetActions = []
        for j, acceptorNetObservation in enumerate(acceptorNetObservationTensors):
            action = self.policy['AcceptorNets'][j].selectAction(acceptorNetObservation)
            acceptorNetActions.append(action)
        
        return offerNetActions, acceptorNetActions
    
    def updateParts(self):
        for acceptorNet in self.policy['AcceptorNets']:
            acceptorNet.update()
        
        for offerNet in self.policy['OfferNets']:
            offerNet.update()
    
    def saveRewards(self,coreChooserRewards,priceChooserRewards,acceptorNetRewards):
        
        for i,reward in enumerate(coreChooserRewards):
            self.policy['OfferNets'][i].coreChooser.buffer.rewards.append(reward[0])
        
        for i,reward in enumerate(priceChooserRewards):
            self.policy['OfferNets'][i].priceChooser.buffer.rewards.append(reward[0])
        
        for i, reward in enumerate(acceptorNetRewards):
            self.policy['AcceptorNets'][i].buffer.rewards.append(reward[0])


class DividedHardcodedAgent(DividedAgent):
    def __init__(self, world):
        super().__init__(world)
        self.policy = {'Acceptors': [HardcodedAcceptor(world) for core in world.cores], 'Offerers': [HardcodedOfferer(world) for slot in range(world.collectionLength)]}
    
    def getActions(self,offerNetObservationTensors,acceptorNetObservationTensors):
        offerNetActions = []
        for i, offerNetObservation in enumerate(offerNetObservationTensors):
            action = self.policy['Offerers'][i].selectAction(offerNetObservation)
            offerNetActions.append(action)
        
        acceptorNetActions = []
        for j, acceptorNetObservation in enumerate(acceptorNetObservationTensors):
            action = self.policy['Acceptors'][j].selectAction(acceptorNetObservation)
            acceptorNetActions.append(action)
        
        return offerNetActions, acceptorNetActions


def numberToNDimensionalAction(number,base,dimensionality):
    '''
    The values of the actions in each dimension range from 0 - (base - 1).
    The number (the input) starts at 0 and ends at (base**dimensionality - 1).
    In this way, a non-negative integer (the input) can be translated into a multidimensional
    plot consisting of array indices.
    '''
    if ((number < 0)|(number >= base **dimensionality)):
        raise ValueError("Illegal Argument")
    number1 = number
    #base spans the n-dimensional hypercube
    dimension = dimensionality - 1
    resultAction = []
    while (len(resultAction) < dimensionality):
        if (dimension == 0):
            actionOfThisDimension = number1
        else:
            actionOfThisDimension = number1 // (base**dimension)
        number1 = number1 - ((base**dimension)*actionOfThisDimension)
        dimension -= 1
        resultAction.append(actionOfThisDimension)
    resultAction.reverse()
    return resultAction

class DividedFixedPriceLocallySharedPPOAgent(DividedAgent):
    def __init__(self,world,env):
        super().__init__(world)
        self.world = world
        self.env = env
        self.locallySharedAcceptorNet = LocallySharedAcceptorPPO(self.world,self.env)
        self.locallySharedOfferNet = LocallySharedOfferPPO(self.world,self.env)
        self.policy = {'AcceptorNet': self.locallySharedAcceptorNet, 'OfferNet': self.locallySharedOfferNet}
        self.CENTRALISATION_SAMPLE = self.env.CENTRALISATION_SAMPLE
    
    def getActions(self,offerNetObservationTensors,acceptorNetObservationTensors):
        offerNetActions = []
        for i, offerNetObservation in enumerate(offerNetObservationTensors):
            action = self.policy['OfferNet'].selectAction(i,offerNetObservation,)
            offerNetActions.append(action)
        
        acceptorNetActions = []
        for j, acceptorNetObservation in enumerate(acceptorNetObservationTensors):
            action = self.policy['AcceptorNet'].selectAction(j,acceptorNetObservation,)
            acceptorNetActions.append(action)
        
        return offerNetActions, acceptorNetActions
    
    def saveRewards(self, offerNetRewards, acceptorNetRewards):
        for i, reward in enumerate(offerNetRewards):
            self.policy['OfferNet'].buffers[i].rewards.append(reward[0])
        
        for i, reward in enumerate(acceptorNetRewards):
            self.policy['AcceptorNet'].buffers[i].rewards.append(reward[0])
    
    def updateParts(self):
        '''
        for subID in range(self.world.numberOfCores):
            self.policy['AcceptorNet'].update(subID)
        
        for subID in range(self.world.collectionLength):
            self.policy['OfferNet'].update(subID)
        '''
        for i in range(self.CENTRALISATION_SAMPLE):
            lastUpdate = False
            randomIndex = random.randint(0,self.world.numberOfCores-1)
            if i == (self.CENTRALISATION_SAMPLE-1):
                lastUpdate = True
            self.policy['AcceptorNet'].update(randomIndex,lastUpdate)
        
        for j in range(self.CENTRALISATION_SAMPLE):
            lastUpdate = False
            randomIndex = random.randint(0,self.world.collectionLength-1)
            if j == (self.CENTRALISATION_SAMPLE-1):
                lastUpdate = True
            self.policy['OfferNet'].update(randomIndex,lastUpdate)
        '''
        for j in range(2):
            lastUpdate = False
            if j == 1:
                lastUpdate = True
            self.policy['OfferNet'].update(j,lastUpdate)
        '''