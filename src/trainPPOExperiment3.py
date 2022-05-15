import time
import statistics
import torch
import numpy as np
from itertools import count
from PPOmodules import *
from world import *
from SchedulingEnvironment import *
from Plot import *
from contextlib import redirect_stdout
import sys, pickle

sys.stdout = sys.__stdout__
original_stdout = sys.stdout

#Setup of this run
PLOTTING = False
fileName = "data/Experiment 3/halb-aggregiert/data{}.pkl"
plotName = 'PPO Training'
plotPath = path = 'C:\\Users\\lenna\\Desktop\\Ausgabe\\'+plotName+' {}.png'
renderingFileName = 'TrainingOutput.txt'
comment = 'Experiment 3., aufgeteilt, Handlungslogging, 8 Kerne'
print(comment)
RENDERING = False


#general settings parameters
dividedAgents = True
hardcodedAgents = False
freePrices = False
aggregatedAgents = False
fullyAggregatedAgents = False
locallySharedParameters = False
globallySharedParameters = False    #if int(sys.argv[1])==1 else False


#Parameters of the world
num_episodes = 2000
episodeLength = 100
numberOfAgents = 1
numberOfCores = 8
rewardMultiplier = 1 
possibleJobPriorities = [3]
possibleJobLengths = [1]
fixPricesList = [1]
probabilities = [1]
meanJobFraction = statistics.mean([F(a,b) for a,b in zip(possibleJobPriorities,possibleJobLengths)])
collectionLength = 8
newJobsPerRoundPerAgent = 8
maxVisibleOffers = 4
acceptorCentralisationFactor = numberOfAgents*numberOfCores if globallySharedParameters else (numberOfCores if locallySharedParameters else 1)
offerNetCentralisationFactor = numberOfAgents*collectionLength if globallySharedParameters else (collectionLength if locallySharedParameters else 1)
assert (len(possibleJobLengths)==len(possibleJobPriorities))
assert sum(probabilities) == 1

world_params_dict = {'num_episodes': num_episodes, 'episodeLength': episodeLength, 'numberOfAgents': numberOfAgents, 'numberOfCores': numberOfCores,\
    'possibleJobPriorities': possibleJobPriorities, 'possibleJobLengths': possibleJobLengths,'collectionLength': collectionLength,'probabilities': probabilities,
    'newJobsPerRoundPerAgent': newJobsPerRoundPerAgent, 'rewardMultiplier': rewardMultiplier, 'freePrices': freePrices, 'fixPricesList': fixPricesList,\
    'maxVisibleOffers': maxVisibleOffers}
world = World(world_params_dict)

#RL Hyperparameters
IS_PPO = True
RANDOMPOLICY = False
LR_ACTOR = 0.003
LR_CRITIC = 0.01
EPS_CLIP = 0.2          
maxJobLength = max(possibleJobLengths)
ACCEPTOR_GAMMA = -((1-maxJobLength)/maxJobLength)+0.04 #0.5 + float(sys.argv[1])*0.05
OFFER_GAMMA = 0.5
RAW_K_EPOCHS = 2   #1 * int(sys.argv[1])
CENTRALISATION_SAMPLE = 2
ACCEPTOR_K_EPOCHS = max(round(RAW_K_EPOCHS/acceptorCentralisationFactor),1)
OFFER_K_EPOCHS = max(round(RAW_K_EPOCHS/offerNetCentralisationFactor),1)
UPDATE_STEP = 2 * episodeLength
netZeroOfferReward = 0.5

RLparamsDict = {'LR_ACTOR': LR_ACTOR,'LR_CRITIC': LR_CRITIC,'EPS_CLIP': EPS_CLIP, 'ACCEPTOR_GAMMA': ACCEPTOR_GAMMA, 'netZeroOfferReward': netZeroOfferReward,
    'OFFER_GAMMA': OFFER_GAMMA, 'RAW_K_EPOCHS': RAW_K_EPOCHS,'RANDOMPOLICY': RANDOMPOLICY, 'UPDATE_STEP': UPDATE_STEP,\
         'globallySharedParameters':globallySharedParameters, 'locallySharedParameters':locallySharedParameters,\
        'ACCEPTOR_K_EPOCHS':ACCEPTOR_K_EPOCHS,'OFFER_K_EPOCHS':OFFER_K_EPOCHS, 'CENTRALISATION_SAMPLE': CENTRALISATION_SAMPLE}


if dividedAgents is True:
    if freePrices is False:
        env = PPODividedFixedPriceEnv(world,RLparamsDict)
    if freePrices is True:
        env = PPODividedFreePriceEnv(world,RLparamsDict)

if aggregatedAgents is True:
    env = PPOAggregatedFixPriceEnv(world,RLparamsDict)

if fullyAggregatedAgents is True:
    env = PPOFullyAggregatedFixPriceEnv(world,RLparamsDict)

if hardcodedAgents is True:
    env = HardcodedFixPriceEnvironment(world,RLparamsDict)

if locallySharedParameters is True:
    env = LocallySharedParamsDividedFixedPriceEnv(world,RLparamsDict)

if globallySharedParameters is True:
    env = GloballySharedParamsDividedFixedPriceEnv(world,RLparamsDict)


parameters = dict(world_params_dict,**RLparamsDict)
parameters['dividedAgents'] = dividedAgents
parameters['hardcodedAgents'] = hardcodedAgents
parameters['aggregatedAgents'] = aggregatedAgents
parameters['fullyAggregatedAgents'] = fullyAggregatedAgents
parameters['freePrices'] = freePrices
parameters['is_PPO'] = True
parameters['comment'] = comment

#Die CoreChooserRewards werden im Fall fixer Preise benutzt.
averageEpisodicCoreChooserRewards = []
averageEpisodicPriceChooserRewards = []
averageEpisodicAcceptorRewards = []
averageEpisodicPrices = []
averageEpisodicAuctioneerReward = []
averageEpisodicDwellTimes = []
averageEpisodicAgentRewards = []
averageEpisodicAcceptionQualities = []
averageEpisodicAcceptionAmounts = []
averageEpisodicTerminationRevenues = []
averageEpisodicTradeRevenues = []

###
averageEpisodicOfferAction0 = []
averageEpisodicOfferAction1 = []
averageEpisodicOfferAction2 = []
averageEpisodicOfferAction3 = []
averageEpisodicOfferAction4 = []
averageEpisodicOfferAction5 = []
averageEpisodicOfferAction6 = []
averageEpisodicOfferAction7 = []
###

time1 = time.time()
# training loop
for i_episode in range(num_episodes):
    time2 = time.time()
    print(i_episode)
    print("Time: {}".format(time2 - time1))
    time1 = time.time()

    newAcceptorObservationTensors,newOfferObservationTensors, newAuctioneerObservation = env.reset()

    if (aggregatedAgents is True)|(fullyAggregatedAgents is True):
        accumulatedCoreChooserReward = torch.tensor([[0] for _ in range(numberOfAgents)])
        accumulatedPriceChooserReward = torch.tensor([[0] for _ in range(numberOfAgents)])
        accumulatedAcceptorReward = torch.tensor([[0] for _ in range(numberOfAgents)])
    else:
        accumulatedCoreChooserReward = torch.tensor([[[0] for _ in range(collectionLength)] for _ in range(numberOfAgents)])
        accumulatedPriceChooserReward = torch.tensor([[[0] for slot in range(collectionLength)] for agent in range(numberOfAgents)],dtype=float)
        accumulatedAcceptorReward = torch.tensor([[[0] for _ in range(numberOfCores)] for _ in range(numberOfAgents)])
    
    env.tradeRevenues = 0 
    env.terminationRevenues = 0   

    accumulatedAgentReward = np.array([0 for _ in range(world.numberOfAgents)])
    prices = []
    collectedAuctioneerReward = []
    collectedAcceptionQualities = []
    collectedAcceptionAmounts = []

    ###
    collectedOfferAction0 = []
    collectedOfferAction1 = []
    collectedOfferAction2 = []
    collectedOfferAction3 = []
    collectedOfferAction4 = []
    collectedOfferAction5 = []
    collectedOfferAction6 = []
    collectedOfferAction7 = []
    ###

    for t in count():
        acceptorActions,offerActions = env.getActionForAllAgents(newAcceptorObservationTensors,newOfferObservationTensors)
        
        collectedOfferAction0.append(offerActions[0][0])
        collectedOfferAction1.append(offerActions[0][1])
        collectedOfferAction2.append(offerActions[0][2])
        collectedOfferAction3.append(offerActions[0][3])
        collectedOfferAction4.append(offerActions[0][4])
        collectedOfferAction5.append(offerActions[0][5])
        collectedOfferAction6.append(offerActions[0][6])
        collectedOfferAction7.append(offerActions[0][7])
        
             
        auctioneer_action = world.auctioneer.getAuctioneerAction(newAuctioneerObservation)

        newAcceptorObservationTensors,newOfferObservationTensors, newAuctioneerObservation,\
        offerRewards, acceptorRewards, auctioneerReward, agentReward, acceptionQuality, done = env.step(offerActions,acceptorActions,auctioneer_action)

        env.saveRewards(offerRewards,acceptorRewards,agentReward)

        if (world.round > 0) & (((world.round) % UPDATE_STEP) == 0) & (RANDOMPOLICY is False):
            env.updateAgents()

        for offer in world.acceptedOffers:
            price = offer.offeredReward
            prices.append((price,offer.jobKind))
        
        if freePrices:
            accumulatedCoreChooserReward += offerRewards[0]
            accumulatedPriceChooserReward += offerRewards[1]
        else:
            accumulatedCoreChooserReward += offerRewards
        accumulatedAgentReward += agentReward
        accumulatedAcceptorReward += acceptorRewards
        collectedAuctioneerReward.append(sum(auctioneerReward.tolist()))
        if acceptionQuality[0] is not None:
            collectedAcceptionQualities.append(acceptionQuality[0])

        collectedAcceptionAmounts.append(acceptionQuality[1])
   
        if RENDERING:
            with open(renderingFileName,'a') as output2:
                sys.stdout = output2
                env.render()
                pprint.pprint("offerNetRewards: {}".format(offerRewards.tolist()))
                pprint.pprint("acceptorNetRewards: {}".format(acceptorRewards.tolist()))
                pprint.pprint("auctioneerRewardPerCore: {}".format(auctioneerReward.tolist()))
                #pprint.pprint("totalAgentRewards: {}".format(totalAgentRewards))
                print("RandomPolicy: {}".format(RANDOMPOLICY))
                sys.stdout = original_stdout
        
        if done:
            acc = []
            for i in range(len(possibleJobPriorities)):
                verweilzeiten = [t.normalisierte_Verweilzeit for t in world.verweilzeiten if (t.Prioritaet==possibleJobPriorities[i])&(t.Bedienzeit == possibleJobLengths[i])]
                if verweilzeiten != []:
                    acc.append(statistics.mean(verweilzeiten))
                else:
                    acc.append(None)
            world.verweilzeiten = []
            averageEpisodicDwellTimes.append(acc)
            acc1 = []
            for i in range(len(possibleJobPriorities)):
                listComp = [tup[0] for tup in prices if tup[1]==i]
                if listComp != []:
                    acc1.append(statistics.mean(listComp))
                else:
                    acc1.append(None)
            averageEpisodicPrices.append(acc1)
            averageEpisodicCoreChooserRewards.append((accumulatedCoreChooserReward / episodeLength).numpy().mean())
            averageEpisodicPriceChooserRewards.append((accumulatedPriceChooserReward / episodeLength).numpy().mean())
            averageEpisodicAcceptorRewards.append((accumulatedAcceptorReward / episodeLength).numpy().mean())
            averageEpisodicAuctioneerReward.append(statistics.mean(collectedAuctioneerReward))
            averageEpisodicAcceptionQualities.append(statistics.mean(collectedAcceptionQualities) if (collectedAcceptionQualities != []) else None)
            averageEpisodicAcceptionAmounts.append(statistics.mean(collectedAcceptionAmounts))
            averageEpisodicAgentRewards.append((accumulatedAgentReward / episodeLength))
            averageEpisodicTradeRevenues.append((env.tradeRevenues / (episodeLength*numberOfAgents*numberOfCores)))
            averageEpisodicTerminationRevenues.append((env.terminationRevenues / (episodeLength*numberOfAgents*numberOfCores)))
            ###
            averageEpisodicOfferAction0.append(statistics.mean(collectedOfferAction0))
            averageEpisodicOfferAction1.append(statistics.mean(collectedOfferAction1))
            averageEpisodicOfferAction2.append(statistics.mean(collectedOfferAction2))
            averageEpisodicOfferAction3.append(statistics.mean(collectedOfferAction3))
            averageEpisodicOfferAction4.append(statistics.mean(collectedOfferAction4))
            averageEpisodicOfferAction5.append(statistics.mean(collectedOfferAction5))
            averageEpisodicOfferAction6.append(statistics.mean(collectedOfferAction6))
            averageEpisodicOfferAction7.append(statistics.mean(collectedOfferAction7))
            ###
            break

argsDict = {}
argsDict['plotPath'] = plotPath
argsDict['acceptorRew'] = averageEpisodicAcceptorRewards
argsDict['coreChooserRew'] = averageEpisodicCoreChooserRewards
argsDict['priceChooserRew'] = averageEpisodicPriceChooserRewards
argsDict['prices'] = averageEpisodicPrices
argsDict['auctioneerRew'] = averageEpisodicAuctioneerReward
argsDict['dwellTimes'] = averageEpisodicDwellTimes
argsDict['meanJob'] = meanJobFraction
argsDict['agentRew'] = averageEpisodicAgentRewards
argsDict['acceptionQuality'] = averageEpisodicAcceptionQualities
argsDict['acceptionAmount']= averageEpisodicAcceptionAmounts
argsDict['terminationRevenues'] = averageEpisodicTerminationRevenues
argsDict['tradeRevenues'] = averageEpisodicTradeRevenues
argsDict['params'] = parameters
###
argsDict['0'] = averageEpisodicOfferAction0
argsDict['1'] = averageEpisodicOfferAction1
argsDict['2'] = averageEpisodicOfferAction2
argsDict['3'] = averageEpisodicOfferAction3
argsDict['4'] = averageEpisodicOfferAction4
argsDict['5'] = averageEpisodicOfferAction5
argsDict['6'] = averageEpisodicOfferAction6
argsDict['7'] = averageEpisodicOfferAction7
###

#Saving the episodic results
while os.path.isfile(fileName.format(i)):
    i += 1
a_file = open(fileName.format(i),"wb")
pickle.dump(argsDict,a_file)
a_file.close()

if PLOTTING is True:
    env.plotResult(argsDict)

env.close()
