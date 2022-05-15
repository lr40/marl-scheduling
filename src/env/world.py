import numpy as np
import random, collections, copy
import torch
from collections import deque, namedtuple
from Agent import *
from Auctioneer import *
import time, statistics
from contextlib import redirect_stdout

Verweilzeit = namedtuple('Verweilzeit',['Prioritaet','Bedienzeit','Verweilzeit','normalisierte_Verweilzeit'])

def resetIDCounter():
    Core.IDCounter = 1
    Job.IDCounter = 1
    Agent.IDCounter = 1
    Offer.offerID = 1
class Core(object):
    IDCounter = 1
    def __init__(self):
        # Core-ID
        self.coreID = Core.IDCounter
        Core.IDCounter += 1
        # the job being executed at the moment. Contains all the relevant information
        self.job = Job(0,0,0,True,None,None)   #The empty job belonging to the Auctioneer
        #ID 0 for the auctioneer
        self.ownerID = 0
    
    def getCoreState(self):
        resultDictionary = {'coreID': self.coreID,'ownerID': self.ownerID, 'jobState': self.job.getJobState()}
        return resultDictionary
    
    def getCoreObservation(self):
        jobState = self.job.getJobState()
        resultDictionary = {'coreID': self.coreID,'ownerID': self.ownerID, 'priority': jobState['priority'], 'remainingLength': jobState['remainingLength']}
        return resultDictionary
    
    def assignCoreToAuctioneer(self):
        self.job = Job(0,0,0,True,None,None)
        self.ownerID = 0
    
    def dispatchNewJobAndReturnOldOne(self, newJob):
        #the old job has to be returned to the accepting party
        if self.job.empty== True:       #Also empty jobs will be returned.
            oldJob = Job(0,0,0,True,None,None)
        else:
            oldJob = self.job
        #assign new job to the core
        self.job = newJob
        if newJob.empty:
            self.ownerID = 0
        else:
            self.ownerID = newJob.ownerID
        return oldJob

class Job(object):
    IDCounter = 1
    def __init__(self,ownerID,priority,initialLength,empty,birthDate,jobKind):
        if empty:
            self.jobID = -1
            self.ownerID = -1
            self.priority = -1
            self.remainingLength = -1
            self.empty = empty
            self.wait = False   # Is initially set to False. If the empty job is a placeholder for a job currently being calculated, the attribute will be set to True.
                                # After terminating the corresponding non-empty job, it is set to True and thus tells the randomized refiller that this slot is available for a randomized refill.
            self.jobKind = -1 
        else:
            self.jobID = Job.IDCounter
            Job.IDCounter += 1
            self.ownerID = ownerID
            self.priority = priority
            self.remainingLength = initialLength
            self.initialLength = initialLength
            self.empty = empty
            self.wait = False   # Will be set to True when an offer is made. Stays this way for one episode and then go back to False.
            self.birthDate = birthDate
            self.jobKind = jobKind

    def getJobState(self):
        return vars(self)
    
    def getJobObservation(self):
        vars1 = vars(self)                              
        resultDictionary = { key: vars1[key] for key in ['priority','remainingLength','empty','wait'] }
        return resultDictionary

class JobCollection(list):
    def __init__(self,collectionLength):
        list.__init__(self,[Job(0,0,0,True,None,None)]*collectionLength)
        self.collectionLength = collectionLength
        self.numberOfFreeSlots = collectionLength
    
    def insertJob(self, job):     
        if self.numberOfFreeSlots > 0:
            for _ in range(self.collectionLength):
                if self[_].empty == True:
                    self[_] = job
                    self.numberOfFreeSlots -= 1
                    break
        else:
            raise "JobCollection is full. The next job couldn't be added.."           
    
    def removeAndReturnEntry(self, queuePosition):
        result = self[queuePosition]
        self[queuePosition] = Job(0,0,0,True,None,None)   #Inserting the empty job such that the list is always filled with jobs.
        self.numberOfFreeSlots += 1
        return result
    
    def getCollectionState(self):
        result = [job.getJobState() for job in self]
        return result
    
    def getcollectionObservation(self):
        result = [job.getJobObservation() for job in self]
        return result 

    def sortByPriority(self):
        self.sort(key=lambda x: x.getJobState()['priority'])

class Offer(object):
    #At each end of a round, this variable has to be reset to 1, in order to keep the observation space low.
    offerID = 1 #Only temporarily valid for one round
    def __init__(self,offererID,recipientID,coreID,queuePosition,jobID,offeredReward,necessaryTime,empty,round1,prio1):
        if empty:
            self.offerID = -1
            self.offererID = -1
            self.recipientID = -1
            self.coreID = -1
            self.queuePosition = -1
            self.jobID = -1
            self.offeredReward = -1
            self.necessaryTime = -1
            self.empty = True
        else:
            self.offerID = Offer.offerID
            Offer.offerID += 1
            self.offererID = offererID
            self.recipientID = recipientID
            self.coreID = coreID
            self.queuePosition = queuePosition
            self.jobID = jobID
            self.prio1 = prio1 #is needed in order to determine rewards for core choosers and price setters
            self.offeredReward = offeredReward
            self.necessaryTime = necessaryTime
            self.jobKind = None
            self.empty = False
            self.round = round1
    
    def getOfferState(self):
        return vars(self)
    
    def getOfferObservation(self):
        vars1 = vars(self)                              
        resultDictionary = { key: vars1[key] for key in ['offerID','offererID','coreID','priority','necessaryTime'] }
        return resultDictionary

class World(object):
    def __init__(self,params):
        self.freePrices = params['freePrices']
        if self.freePrices is False:
            self.listOfFixPrices = params['fixPricesList']
        self.numberOfAgents = params['numberOfAgents']
        self.numberOfCores = params['numberOfCores']
        self.possibleJobLengths = params['possibleJobLengths']
        self.possibleJobPriorities = params['possibleJobPriorities']
        self.probabilities = params['probabilities']
        self.accProbabilities = [sum(self.probabilities[:(i+1)]) for i in range(len(self.probabilities))]
        #Stimmt nicht!
        self.maxSumToOffer = max(self.possibleJobPriorities)
        self.collectionLength = params['collectionLength']
        self.maxAmountOfOffers = self.numberOfAgents * self.collectionLength
        self.maxAmountOfOffersToOneAgent = (self.numberOfAgents) * self.collectionLength  #You do shake hands with yourself to reduce complexity.
        self.maxAmountOfAcceptionsPerTimeStepPerAgent = min(self.maxAmountOfOffersToOneAgent,self.numberOfCores)
        self.offers = []     
        self.newJobsPerRoundPerAgent = params['newJobsPerRoundPerAgent']
        
        self.liabilityList = [deque([]) for _ in range(self.numberOfCores)]
        #                               Kern-ID,        an agentID,    von agentID,      offeredReward, necessaryTime
        self.jobTerminationInfo = []
        self.verweilzeiten = []
        self.round = 0
        self.episodeLength = params['episodeLength']
        self.maxVisibleOffers = params['maxVisibleOffers']
        self.acceptedOffers = []
        self.rewardMultiplier = params['rewardMultiplier']
        self.cores = [Core() for i in range(self.numberOfCores)]
        self.agents = None
        self.randomPolicy = False
        self.acceptedOfferRatios = [[[] for _ in range(self.numberOfCores)] for _ in range(self.numberOfAgents)]
        self.auctioneer = Auctioneer(self)
        self.auctioneer.updateOwnedCores(self)
    
    def resetLiabilityListForACore(self,coreID):  #-1, because index and ID are shifted by one, +1, because the auctioneer is added to it
        self.liabilityList[coreID -1] = deque([]) 
    
    def executeAnOffer(self,offerID):
        offer = next((x for x in self.offers if x.offerID==offerID))
        core = next((x for x in self.cores if x.coreID == offer.coreID))
        '''
        It could be that the ownership status has changed in the meantime. That the agent has accepted an offer for
        a core for which he no longer has any ownership rights.
        '''
        if (offer.recipientID == core.ownerID):
            queuePosition = offer.queuePosition
            if offer.recipientID == 0:
                recipient = self.auctioneer
                recipientID = self.auctioneer.auctioneerID
            else:
                recipient = next((x for x in self.agents if x.agentID==offer.recipientID))
                recipientID = recipient.agentID
            offerer = next((x for x in self.agents if x.agentID==offer.offererID))
            newJob = offerer.collection.removeAndReturnEntry(queuePosition)
            newJob.wait = False   # Will be reset. It should be fresh when it is returned.
            oldJob = core.dispatchNewJobAndReturnOldOne(newJob)
            if recipientID != 0:
                recipient.collection.insertJob(oldJob) #The old job is returned to the accepting party. The random generator must ensure that there is room for it.
                self.acceptedOfferRatios[recipientID - 1][core.coreID -1].append([offer.offeredReward,offer.necessaryTime])
            liabilityListEntry = copy.deepcopy(offer)
            liabilityListEntry.round = self.round
            self.liabilityList[core.coreID-1].appendleft(liabilityListEntry)
                                                                                           
            recipient.updateOwnedCores(self)
            offerer.updateOwnedCores(self)
            self.acceptedOffers.append(copy.deepcopy(offer))
    
    def step1(self,allOfferNetActionTuples,allAcceptorNetsActionTensors,correspondingOfferIDs,auctioneer_action, auctioneer_correspondingOfferIDs):
        '''
        Formats of the transferred aggregated actions and information:
        allAcceptorNetsActionTensors: [array1([Index1, Index2, ..., IndexNumCores]),...,arrayNumAgents([Index1, Index2, ..., IndexNumCores])]
        allOfferNetActionTuples: [array1([Offer-CoreID1,...,Offer-CoreIDCollLen]),...,arrayNumAgents([Offer-CoreID1,...,Offer-CoreIDCollLen])]
        correspondingOfferIDs: [list1[array1([OfferID1,...,OfferIDMaxAmountOffers]),...,arrayNumCores([OfferID1,...,OfferIDMaxAmountOffers]],...,listNumAgents[[OfferID1,...,OfferIDMaxAmountOffers]),...,arrayNumCores([OfferID1,...,OfferIDMaxAmountOffers]]]
        '''
        for core,owner,jobID,generatedReward,round1 in self.jobTerminationInfo:
            self.resetLiabilityListForACore(core.coreID)
        
        self.jobTerminationInfo = []
        self.acceptedOffers = []
        
        self.executeAgentAcceptions1(allAcceptorNetsActionTensors,correspondingOfferIDs) #comment out to disable intra-agent trading
        
        self.executeAuctioneerAcceptions(auctioneer_action,auctioneer_correspondingOfferIDs)
        
        self.processOneTimestepAndUpdateOwnership()
        
        #Reset the offers of the last round
        self.offers = []
        Offer.offerID = 1

        if self.freePrices is False:
            self.createFixPriceOfferObjectsFromActions(allOfferNetActionTuples)
        else:
            self.createFreePriceOfferObjectsFromActions(allOfferNetActionTuples)

        self.fillQueuesWithNewRandomJobs()

        self.round += 1
            
    def processOneTimestepAndUpdateOwnership(self):
        for core in self.cores:
            if core.job.empty == False:
                core.job.remainingLength -= 1
                if core.job.remainingLength == 0:
                    
                    generatedReward = self.rewardMultiplier * core.job.priority
                    jobID = core.job.jobID
                    owner = next((x for x in self.agents if x.agentID == core.ownerID))
                    ownerID = owner.agentID
                    self.jobTerminationInfo.append(copy.deepcopy((core,ownerID,jobID,generatedReward,self.round+1)))
                                    # -1 to avoid an off-by-one error. Job is generated at the end of the old round with the old round number as birthdate.
                    self.verweilzeiten.append(Verweilzeit(core.job.priority,core.job.initialLength,self.round - core.job.birthDate,(self.round - core.job.birthDate - 1)/core.job.initialLength))
                    core.assignCoreToAuctioneer()
        #This whole owned core list is really just for testing. It may be unnecessary.
        self.auctioneer.updateOwnedCores(self)
        for agent in self.agents:
            agent.updateOwnedCores(self)
        
        for agentEntry in self.acceptedOfferRatios:
            for acceptorEntry in agentEntry:
                for tradeEntry in acceptorEntry:
                    tradeEntry[1] -= 1
        
    def fillQueuesWithNewRandomJobs(self):
        for agent in self.agents:
            if (len(agent.ownedCores) + self.newJobsPerRoundPerAgent) <= agent.collection.numberOfFreeSlots:
                agent.fillCollectionRandomly(amountOfNewEntries=self.newJobsPerRoundPerAgent)
                '''
                This ensures that the collection can never overflow. 
                For example, if you were to sell all of your own cores and then a new job was added.
                '''
    def executeAuctioneerAcceptions(self,auctioneerAction,auctioneer_correspondingOfferIDs):

        for j, chosenIndexValue in enumerate(auctioneerAction):
            '''Because if the chosen index is greater by one (see below), no offer was accepted on purpose. 
            This action should be equivalent to accepting an empty offer.'''
            if chosenIndexValue < len(auctioneer_correspondingOfferIDs[j]):
                offerIDToBeExecuted = auctioneer_correspondingOfferIDs[j][chosenIndexValue]
                #Der Auktionator könnte als lernende Einheit auch nicht-vorhandene ("negative") Offer-IDs annehmen.
                if 0 < offerIDToBeExecuted:
                    self.executeAnOffer(offerIDToBeExecuted)
            else:
                assert (chosenIndexValue == len(auctioneer_correspondingOfferIDs[j]))
    

    def executeAgentAcceptions1(self,allAcceptorNetsActionTensors,correspondingOfferIDs):
        for i,agent in enumerate(self.agents):
            offerIDs = correspondingOfferIDs[i]
            chosenIndexValues = allAcceptorNetsActionTensors[i]
            for j, chosenIndexValue in enumerate(chosenIndexValues):
                '''Because if the chosen index is greater by one (see below), no offer was accepted on purpose. 
                This action should be equivalent to accepting an empty offer.'''
                if chosenIndexValue < len(offerIDs[j]):
                    offerIDToBeExecuted = offerIDs[j][int(chosenIndexValue)]
                    if 0 < offerIDToBeExecuted:
                        self.executeAnOffer(offerIDToBeExecuted)
                else:
                    assert (chosenIndexValue == len(offerIDs[j]))
    
    def createFixPriceOfferObjectsFromActions(self,allOfferNetActionLists):

        for i, agent in enumerate(self.agents):
            offeringAgent = self.agents[i]
            actionList = allOfferNetActionLists[i]
            for j, action in enumerate(actionList):
                coreID = action + 1
                correspondingCore = next((x for x in self.cores if x.coreID==coreID), None)
                slot = j
                correspondingJob = offeringAgent.collection[slot]
                #progress = correspondingJob.remainingLength / self.possibleJobLengths[correspondingJob.jobKind]
                fixReward = self.listOfFixPrices[correspondingJob.jobKind]
                offeredReward = fixReward #round(fixReward*progress)
                priority = correspondingJob.priority
                necessaryTime = correspondingJob.remainingLength
                if (correspondingCore is not None) & (correspondingJob.empty == False) & (correspondingJob.wait == False):
                    necessaryTime = correspondingJob.remainingLength
                    offerObject = Offer(offeringAgent.agentID,correspondingCore.ownerID,correspondingCore.coreID,slot,correspondingJob.jobID,offeredReward,necessaryTime,False,self.round,correspondingJob.priority)
                    offerObject.jobKind = correspondingJob.jobKind
                    self.offers.append(offerObject)
                    correspondingJob.wait = True
                else:
                    correspondingJob.wait = False 
    
    def createFreePriceOfferObjectsFromActions(self,allOfferNetActionLists):
        for i, agent in enumerate(self.agents):
            offeringAgent = self.agents[i]
            actionList = allOfferNetActionLists[i]
            for j, action in enumerate(actionList):
                coreID = (action[0] + 1)
                correspondingCore = next((x for x in self.cores if x.coreID==coreID), None)
                offeredReward = action[1]
                slot = j
                correspondingJob = offeringAgent.collection[slot]
                necessaryTime = correspondingJob.remainingLength
                if (correspondingCore is not None) & (correspondingJob.empty == False) & (correspondingJob.wait == False):
                    necessaryTime = correspondingJob.remainingLength
                    offerObject = Offer(offeringAgent.agentID,correspondingCore.ownerID,correspondingCore.coreID,slot,correspondingJob.jobID,offeredReward,necessaryTime,False,self.round,correspondingJob.priority)
                    offerObject.jobKind = correspondingJob.jobKind
                    self.offers.append(offerObject)
                    correspondingJob.wait = True
                else:
                    correspondingJob.wait = False 

        

    
    
    




        
        
        

    

        
