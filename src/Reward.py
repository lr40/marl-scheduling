import numpy as np
import copy

def getDividedFreePricesReward(env,commercialFreePriceReward):
    #calculate the offerNetRewards
    priceChooserRewards = np.array([[[0] for slot in range(env.world.collectionLength)] for agent in range(env.world.numberOfAgents)],dtype=float)
    coreChooserRewards = np.array([[[0] for slot in range(env.world.collectionLength)] for agent in range(env.world.numberOfAgents)],dtype=float)
    if commercialFreePriceReward is True:
        for offer in env.world.acceptedOffers:
            offerer = next((x for x in env.world.agents if offer.offererID==x.agentID))
            slotID = offer.queuePosition
            coreChooserReward = offer.prio1 #(offer.prio1 / offer.necessaryTime) - ((env.formerCorePrios[offer.coreID - 1] / env.formerCoreLengths[offer.coreID - 1]) if (env.formerCorePrios[offer.coreID - 1] != -1) else 0)
            priceChooserReward = env.netZeroOfferReward if ((offer.prio1 - offer.offeredReward) == 0) else (offer.prio1 - offer.offeredReward)
            priceChooserRewards[offerer.agentID - 1][slotID] = priceChooserReward
            coreChooserRewards[offerer.agentID - 1][slotID] = coreChooserReward
    if commercialFreePriceReward is False:
        for offer in env.world.acceptedOffers:
            offerer = next((x for x in env.world.agents if offer.offererID==x.agentID))
            slotID = offer.queuePosition
            coreChooserReward = offer.prio1 #(offer.prio1 / offer.necessaryTime) - ((env.formerCorePrios[offer.coreID - 1] / env.formerCoreLengths[offer.coreID - 1]) if (env.formerCorePrios[offer.coreID - 1] != -1) else 0)
            priceChooserReward = offer.prio1 if ((offer.prio1 - offer.offeredReward) >= 0) else (offer.prio1 - offer.offeredReward)
            priceChooserRewards[offerer.agentID - 1][slotID] = priceChooserReward
            coreChooserRewards[offerer.agentID - 1][slotID] = coreChooserReward
        

    auctioneerReward = np.array([0 for core in env.world.cores])
    agentReward = np.array([0 for agent in env.world.agents])
    acceptorNetRewards = np.array([[[0] for core in range(env.world.numberOfCores)] for agent in range(env.world.numberOfAgents)])
    for core, ownerID, jobID, generatedReward, timeStamp in env.world.jobTerminationInfo:
        # Process rewardchains for each core 
        rewardChainList = env.world.liabilityList[core.coreID - 1]
        
        acceptorNetRewards[ownerID - 1][core.coreID - 1] = generatedReward
        lastTimeStamp = timeStamp
        timeMeasure = 0
        for entry in rewardChainList:
            timeMeasure += (lastTimeStamp - entry.round)
            lastTimeStamp = entry.round
            
            negotiatedRewardRatio = entry.offeredReward / entry.necessaryTime
            tradedReward = round(negotiatedRewardRatio * timeMeasure)
            #Die untere Zeile war leider auskommentiert
            acceptorNetRewards[entry.offererID - 1][core.coreID - 1] -= tradedReward
            agentReward[entry.offererID - 1] -= tradedReward
            # Auktionator sollte unbedingt noch mit eingebaut werden!
            if (entry.recipientID > 0):
                acceptorNetRewards[entry.recipientID - 1][core.coreID - 1] +=  tradedReward
                agentReward[entry.recipientID - 1] += tradedReward
            if (entry.recipientID == 0):
                auctioneerReward[core.coreID - 1] = tradedReward
        
        env.world.resetLiabilityListForACore(core.coreID)
    
    return (coreChooserRewards, priceChooserRewards), acceptorNetRewards, auctioneerReward, agentReward
        

def getAggregatedFixedPricesReward(env):
    offerRewards = np.array([[0] for agent in range(env.world.numberOfAgents)])
    acceptorRewards = np.array([[0] for agent in range(env.world.numberOfAgents)])
    for offer in env.world.acceptedOffers:
        #reward = (offer.prio1 / offer.necessaryTime) - ((env.formerCorePrios[offer.coreID - 1] / env.formerCoreLengths[offer.coreID - 1]) if (env.formerCorePrios[offer.coreID - 1] != -1) else 0)
        reward = offer.prio1
        aggregatedAgentID = offer.offererID
        slotID = offer.queuePosition
        # set the Reward for the offerNet
        offerRewards[aggregatedAgentID - 1][0] += reward
    
    #Sichere Kopie zum Iterieren nötig, da wir Elemente entfernen.
    safeCopy = copy.deepcopy(env.world.acceptedOfferRatios)
    for i, agentEntry in enumerate(safeCopy):
        for j, acceptorEntry in enumerate(agentEntry):
            for  tradeEntry in acceptorEntry:
                if tradeEntry[1] == 0:
                    #acceptorRewards[i] += tradeEntry[0]
                    #env.tradeRevenues += tradeEntry[0]
                    env.world.acceptedOfferRatios[i][j].remove(tradeEntry)
                if tradeEntry[1] < 0:
                    acceptorEntry.remove(tradeEntry)
    
    auctioneerReward = np.array([0 for core in env.world.cores])
    agentReward = np.array([0 for agent in env.world.agents])
    #calculate the acceptorNetRewards
    for core, ownerID, jobID, generatedReward, timeStamp in env.world.jobTerminationInfo:
        # Process rewardchains for each core             
        rewardChainList = env.world.liabilityList[core.coreID - 1]
        
        acceptorRewards[ownerID - 1] += generatedReward
        agentReward[ownerID - 1] += generatedReward
        lastTimeStamp = timeStamp
        timeMeasure = 0
        for entry in rewardChainList:
            timeMeasure += (lastTimeStamp - entry.round)
            lastTimeStamp = entry.round
            
            negotiatedRewardRatio = entry.offeredReward / entry.necessaryTime
            tradedReward = round(negotiatedRewardRatio * timeMeasure)
            #Die untere Zeile war leider auskommentiert.
            acceptorRewards[entry.offererID - 1] -= tradedReward
            agentReward[entry.offererID - 1] -= tradedReward
            # Auktionator sollte unbedingt noch mit eingebaut werden!
            if (entry.recipientID > 0):
                agentReward[entry.recipientID - 1] += tradedReward
            if (entry.recipientID == 0):
                auctioneerReward[core.coreID - 1] = tradedReward
        
        env.world.resetLiabilityListForACore(core.coreID)
    
    return offerRewards, acceptorRewards, auctioneerReward, agentReward

def getDividedFixedPricesReward(env):
    '''
    Der Reward soll für die Akzeptoren zuverlässiger erfolgen und nicht mehr den Wirren der Handlungen nachfolgender Akzeptoren unterworfen sein.
    Denn die unterschiedlich langen Reward-Chains können eine Ursache für ein störhaftes Reward-Signal sein.
    Reward entsteht also entweder durch Abarbeitung eines eigenen Jobs oder durch Annahme eines Angebots und Abwarten der Zeitspanne.
    '''
    acceptorNetRewards = np.array([[[0] for core in range(env.world.numberOfCores)] for agent in range(env.world.numberOfAgents)])
    offerNetRewards = np.array([[[0] for slot in range(env.world.collectionLength)] for agent in range(env.world.numberOfAgents)])
    for offer in env.world.acceptedOffers:
        #reward = (offer.prio1 / offer.necessaryTime) - ((env.formerCorePrios[offer.coreID - 1] / env.formerCoreLengths[offer.coreID - 1]) if (env.formerCorePrios[offer.coreID - 1] != -1) else 0)
        reward = offer.prio1
        dividedAgentID = offer.offererID
        coreID = offer.coreID
        slotID = offer.queuePosition
        offerNetRewards[dividedAgentID - 1][slotID] = reward
    
    #Sichere Kopie zum Iterieren nötig, da wir Elemente entfernen.
    safeCopy = copy.deepcopy(env.world.acceptedOfferRatios)
    for i, agentEntry in enumerate(safeCopy):
        for j, acceptorEntry in enumerate(agentEntry):
            for  tradeEntry in acceptorEntry:
                if tradeEntry[1] == 0:
                    #acceptorNetRewards[i][j] += tradeEntry[0]
                    #env.tradeRevenues += tradeEntry[0]
                    env.world.acceptedOfferRatios[i][j].remove(tradeEntry)
                if tradeEntry[1] < 0:
                    acceptorEntry.remove(tradeEntry)   
    
    auctioneerReward = np.array([0 for core in env.world.cores])
    agentReward = np.array([0 for agent in env.world.agents])
    #calculate the acceptorNetRewards
    for core, ownerID, jobID, generatedReward, timeStamp in env.world.jobTerminationInfo:
        # Process rewardchains for each core             
        rewardChainList = env.world.liabilityList[core.coreID - 1]
        # Hier ist das absolute Gleichsetzen gerechtfertigt, weil ein aufgeteilter Akzeptor nur einen Job je Runde beenden kann.
        acceptorNetRewards[ownerID - 1][core.coreID - 1] = generatedReward
        agentReward[ownerID - 1] += generatedReward
        env.terminationRevenues += generatedReward
        lastTimeStamp = timeStamp
        timeMeasure = 0
        for entry in rewardChainList:
            timeMeasure += (lastTimeStamp - entry.round)
            lastTimeStamp = entry.round
            
            negotiatedRewardRatio = entry.offeredReward / entry.necessaryTime
            tradedReward = round(negotiatedRewardRatio * timeMeasure)
            acceptorNetRewards[entry.offererID - 1][core.coreID - 1] -= tradedReward
            agentReward[entry.offererID - 1] -= tradedReward
            if (entry.recipientID > 0):
                agentReward[entry.recipientID - 1] += tradedReward
                acceptorNetRewards[entry.recipientID - 1][core.coreID - 1] += tradedReward
            if (entry.recipientID == 0):
                auctioneerReward[core.coreID - 1] = tradedReward
        
        env.world.resetLiabilityListForACore(core.coreID)
        
    return offerNetRewards, acceptorNetRewards, auctioneerReward, agentReward