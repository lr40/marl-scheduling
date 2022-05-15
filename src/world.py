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
        #core muss noch gefunden werden
        if self.job.empty== True:       #Diese Fallunterscheidung ist nur der Übersicht halber, damit klar ist, dass auch leere Jobs retourniert werden.
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
            self.wait = False   # Wird erstmal auf False gesetzt. Falls der leere Job noch ein Platzhalter für einen gerade in der Berechnung steckenden Job ist
            self.jobKind = -1   # bleibt das Attribut auf True. Nach Terminierung dieses Jobs, wird es auf True gesetzt und signalisiert damit dem randomisierten Nachfüller,
                                # dass dieser Slot für eine randomisierte Nachfüllung zur Verfügung steht.
        else:
            self.jobID = Job.IDCounter
            Job.IDCounter += 1
            self.ownerID = ownerID
            self.priority = priority
            self.remainingLength = initialLength
            self.initialLength = initialLength
            self.empty = empty
            self.wait = False   #Wird auf True gesetzt, wenn ein Angebot gemacht wurde. Sollte für eine Episode so bleiben und dann wieder auf False gehen
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
    
    def insertJob(self, job):      #Bewirkt, dass der erste freie Platz befüllt wird, aber dass andere Einträge keinesfalls verschoben werden.
        if self.numberOfFreeSlots > 0:
            for _ in range(self.collectionLength):
                if self[_].empty == True:
                    self[_] = job
                    self.numberOfFreeSlots -= 1
                    break
        else:
            raise "Die JobCollection ist voll. Ein weiterer Job konnte nicht hinzugefügt werden."           
    
    def removeAndReturnEntry(self, queuePosition):
        result = self[queuePosition]
        self[queuePosition] = Job(0,0,0,True,None,None)   #Einfügen des leeren Jobs, damit die Liste immer mit Jobs befüllt ist.
        self.numberOfFreeSlots += 1
        return result
    
    def getCollectionState(self):
        result = [job.getJobState() for job in self]
        return result
    
    def getcollectionObservation(self):
        result = [job.getJobObservation() for job in self]
        return result  #Hatte hier auch mal mit list(enumerate(result)) gearbeitet. Aber der Listenplatz ist ja schon implizit durch die Ordnung gegeben.
                    # Wie bei den anderen Observations auch.

    def sortByPriority(self):
        self.sort(key=lambda x: x.getJobState()['priority'])

class Offer(object):
    #Diese Variable muss bei jedem Rundenende wieder auf eins zurückgesetzt werden, um den Beobachtungsraum klein zu halten.
    offerID = 1 #Sind nur temporär gültig für eine Episode
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
            self.prio1 = prio1 # Dient nur der Information, um den Reward für die Kern- und Preissetzer bestimmen zu können
            self.offeredReward = offeredReward
            self.necessaryTime = necessaryTime
            self.jobKind = None
            self.empty = False
            self.round = round1
    
    def getOfferState(self):
        return vars(self)
    
    def getOfferObservation(self):
        vars1 = vars(self)                              #offererID könnte noch wichtig werden für den aktienbasierten Ansatz
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
        self.offers = []     #Hier waren früher leere Angebote drin. Die müssen ja aber eigentlich nur Teil der Beobachtung sein.
        self.newJobsPerRoundPerAgent = params['newJobsPerRoundPerAgent']
        #Refactoring: Die beiden folgenden dienen eher der Abrechnung und könnten daher auch Teil der Environment sein.
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
    
    def resetLiabilityListForACore(self,coreID):  #-1, weil Index und ID um eins verschoben sind, +1, weil ja noch der Auktionator dazu kommt
        self.liabilityList[coreID -1] = deque([]) 
    
    def executeAnOffer(self,offerID):
        offer = next((x for x in self.offers if x.offerID==offerID))
        core = next((x for x in self.cores if x.coreID == offer.coreID))
        #Es könnte ja sein, dass sich der Besitzstatus in der Zwischenzeit geändert hat. Dass der Agent ein Angebot für einen Kern angenommen hat, für
        # den er gar keine Besitzrechte mehr hat.
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
            newJob.wait = False   # Wird zurückgesetzt. Beim Zurückgeben soll er ja frisch sein.
            oldJob = core.dispatchNewJobAndReturnOldOne(newJob)
            if recipientID != 0:
                recipient.collection.insertJob(oldJob) #Der alte Job wird wieder an die annehmende Partei zurückgegeben. Dass dafür auch Platz ist muss der Zufallsgenerator sicherstellen.
                self.acceptedOfferRatios[recipientID - 1][core.coreID -1].append([offer.offeredReward,offer.necessaryTime])
            liabilityListEntry = copy.deepcopy(offer)
            liabilityListEntry.round = self.round
            self.liabilityList[core.coreID-1].appendleft(liabilityListEntry)    # -1 von der CoreID;
                                                                                           
            recipient.updateOwnedCores(self)
            offerer.updateOwnedCores(self)
            self.acceptedOffers.append(copy.deepcopy(offer))
    
    def step1(self,allOfferNetActionTuples,allAcceptorNetsActionTensors,correspondingOfferIDs,auctioneer_action, auctioneer_correspondingOfferIDs):
        '''
        Formate der übergebenen, aggregierten Handlungen und Informationen:
        allAcceptorNetsActionTensors: [array1([Index1, Index2, ..., IndexNumCores]),...,arrayNumAgents([Index1, Index2, ..., IndexNumCores])]
        allOfferNetActionTuples: [array1([Offer-CoreID1,...,Offer-CoreIDCollLen]),...,arrayNumAgents([Offer-CoreID1,...,Offer-CoreIDCollLen])]
        correspondingOfferIDs: [list1[array1([OfferID1,...,OfferIDMaxAmountOffers]),...,arrayNumCores([OfferID1,...,OfferIDMaxAmountOffers]],...,listNumAgents[[OfferID1,...,OfferIDMaxAmountOffers]),...,arrayNumCores([OfferID1,...,OfferIDMaxAmountOffers]]]
        '''
        for core,owner,jobID,generatedReward,round1 in self.jobTerminationInfo:
            self.resetLiabilityListForACore(core.coreID)
        
        self.jobTerminationInfo = []
        self.acceptedOffers = []
        
        self.executeAgentAcceptions1(allAcceptorNetsActionTensors,correspondingOfferIDs) #auskommentieren um die Kein-Trading-Benchmark zu erstellen
        
        self.executeAuctioneerAcceptions(auctioneer_action,auctioneer_correspondingOfferIDs)
        
        self.processOneTimestepAndUpdateOwnership()
        
        #Rücksetzen der Angebote der letzten Runde
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
                    #Durch die doppelte Entlohnung kann der Reward einfach fair geteilt werden. Es muss kein Preis mehr festgelegt werden.
                    generatedReward = self.rewardMultiplier * core.job.priority
                    jobID = core.job.jobID
                    owner = next((x for x in self.agents if x.agentID == core.ownerID))
                    ownerID = owner.agentID
                    self.jobTerminationInfo.append(copy.deepcopy((core,ownerID,jobID,generatedReward,self.round+1)))
                                                                            # -1, um einen Off-By-One Fehler zu vermeiden. Job wird ja am Ende der alten Runde generiert mit der alten Rundennummer als Birthdate
                    self.verweilzeiten.append(Verweilzeit(core.job.priority,core.job.initialLength,self.round - core.job.birthDate,(self.round - core.job.birthDate - 1)/core.job.initialLength))
                    core.assignCoreToAuctioneer()
        #Diese ganze owned Core Liste ist eigentlich nur zum Testen. Eventuell ist sie unnötig.
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
                #So wird sichergestellt, dass die collection niemals überlaufen kann. Zum Beispiel, wenn man alle eigenen Kerne verkaufen würde und
                # dann noch ein neuer Job hinzukäme.
    
    def executeAuctioneerAcceptions(self,auctioneerAction,auctioneer_correspondingOfferIDs):

        for j, chosenIndexValue in enumerate(auctioneerAction):
                #Denn wenn der gewählte Index um eins größer ist (s.u.), wurde mit Absicht kein Angebot angenommen.
                #Diese Handlung dürfte äquivalent sein zur Annahme eines leeren Angebots.
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
                #Denn wenn der gewählte Index um eins größer ist (s.u.), wurde mit Absicht kein Angebot angenommen.
                #Diese Handlung dürfte äquivalent sein zur Annahme eines leeren Angebots.
                if chosenIndexValue < len(offerIDs[j]):
                    offerIDToBeExecuted = offerIDs[j][int(chosenIndexValue)]
                    #Bei möglichen Fehlern siehe oben
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

        

    
    
    




        
        
        

    

        
