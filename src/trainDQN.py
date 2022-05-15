import time
import statistics
import torch
import numpy as np
from itertools import count
from PPOmodules import *
from world import *
from SchedulingEnvironment import *
from Plot import *
import sys, pickle

sys.stdout = sys.__stdout__
original_stdout = sys.stdout

#Setup of this run
PLOTTING = False
fileName = "data/Test1/data{}.pkl"
plotName = 'PPO Training'
plotPath = path = 'C:\\Users\\lenna\\Desktop\\Plots3\\'+plotName+' {}.png'
renderingFileName = 'TrainingOutput.txt'
comment = 'DQN Experiment Einführung'
RENDERING = False

#general settings parameters
hardcodedAgents = False
freePrices = False
aggregatedAgents = False
locallySharedParameters = False
globallySharedParameters = False 


#Parameters of the world
num_episodes = 6000
episodeLength = 100
numberOfAgents = 2
numberOfCores = 2
rewardMultiplier = 1 
possibleJobPriorities = [3,10]
possibleJobLengths = [6,3]
fixPricesList = [2,7]
probabilities = [0.8,0.2]
meanJobFraction = statistics.mean([F(a,b) for a,b in zip(possibleJobPriorities,possibleJobLengths)])
collectionLength = 3
newJobsPerRoundPerAgent = 1
maxVisibleOffers = 4
assert (len(possibleJobLengths)==len(possibleJobPriorities))
assert sum(probabilities) == 1

world_params_dict = {'num_episodes': num_episodes, 'episodeLength': episodeLength, 'numberOfAgents': numberOfAgents, 'numberOfCores': numberOfCores,\
    'possibleJobPriorities': possibleJobPriorities, 'possibleJobLengths': possibleJobLengths,'collectionLength': collectionLength,'probabilities': probabilities,
    'newJobsPerRoundPerAgent': newJobsPerRoundPerAgent, 'rewardMultiplier': rewardMultiplier, 'freePrices': freePrices, 'fixPricesList': fixPricesList,\
    'maxVisibleOffers': maxVisibleOffers}
world = World(world_params_dict)

#RL Hyperparameters
IS_DQN = True
RANDOMPOLICY = False
BATCH_SIZE = 10
OFFER_GAMMA = 0.5
maxJobLength = max(possibleJobLengths)
ACCEPTOR_GAMMA = -((1-maxJobLength)/maxJobLength)+0.04
ACCEPTOR_GAMMA = 0.84
RUN_START = 0.9
RUN_END = 0.05
RUN_DECAY = 500
REPLAY_MEMORY_SIZE = 5000
TARGET_UPDATE = 2
netZeroOfferReward = 0.5

RLparamsDict = {'BATCH_SIZE': BATCH_SIZE,'OFFER_GAMMA': OFFER_GAMMA,'ACCEPTOR_GAMMA': ACCEPTOR_GAMMA, 'RUN_START': RUN_START, 'REPLAY_MEMORY_SIZE': REPLAY_MEMORY_SIZE,\
     'RUN_END': RUN_END,'RUN_DECAY': RUN_DECAY, 'TARGET_UPDATE': TARGET_UPDATE, 'RANDOMPOLICY': RANDOMPOLICY, 'IS_DQN': IS_DQN,'freePrices': False, \
         'netZeroOfferReward': netZeroOfferReward}

env = DQNDividedFixedPricesEnv(world,RLparamsDict)


parameters = dict(world_params_dict,**RLparamsDict)
parameters['hardcodedAgents'] = hardcodedAgents
parameters['aggregatedAgents'] = aggregatedAgents
parameters['freePrices'] = freePrices
parameters['is_DQN'] = True
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

time1 = time.time()
# training loop
for i_episode in range(num_episodes):
    time2 = time.time()
    print(i_episode)
    print("Time: {}".format(time2 - time1))
    time1 = time.time()

    newAcceptorObservationTensors,newOfferObservationTensors, newAuctioneerObservation = env.reset()

    if aggregatedAgents is False:
        accumulatedCoreChooserReward = torch.tensor([[[0] for _ in range(collectionLength)] for _ in range(numberOfAgents)])
        accumulatedPriceChooserReward = torch.tensor([[[0] for slot in range(collectionLength)] for agent in range(numberOfAgents)],dtype=float)
        accumulatedAcceptorReward = torch.tensor([[[0] for _ in range(numberOfCores)] for _ in range(numberOfAgents)])
    else:
        accumulatedCoreChooserReward = torch.tensor([[0] for _ in range(numberOfAgents)])
        accumulatedPriceChooserReward = torch.tensor([[0] for _ in range(numberOfAgents)])
        accumulatedAcceptorReward = torch.tensor([[0] for _ in range(numberOfAgents)])

    env.tradeRevenues = 0 
    env.terminationRevenues = 0

    accumulatedAgentReward = np.array([0 for _ in range(world.numberOfAgents)])
    prices = []
    collectedAuctioneerReward = []
    collectedAcceptionQualities = []
    collectedAcceptionAmounts = []

    for t in count():
        acceptorActions,offerActions = env.getActionForAllAgents(newAcceptorObservationTensors,newOfferObservationTensors)
        
        auctioneer_action = world.auctioneer.getAuctioneerAction(newAuctioneerObservation)

        oldAcceptorObservationTensors = newAcceptorObservationTensors
        env.oldAcceptorObservationTensors = oldAcceptorObservationTensors
        oldOfferObservationTensors = newOfferObservationTensors
        env.oldOfferObservationTensors = oldOfferObservationTensors
     
        newAcceptorObservationTensors,newOfferObservationTensors, newAuctioneerObservation,\
        offerRewards, acceptorRewards, auctioneerReward, agentReward, acceptionQuality, done = env.step(offerActions,acceptorActions,auctioneer_action)

        for offer in world.acceptedOffers:
            price = offer.offeredReward
            prices.append((price,offer.jobKind))
        
        env.newAcceptorObservationTensors = newAcceptorObservationTensors
        env.newOfferObservationTensors = newOfferObservationTensors
        
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
            break

        if RANDOMPOLICY is False:
            env.updateAcceptorMemoriesAndOptimize(acceptorActions,acceptorRewards)
            env.updateOfferMemoriesAndOptimize(offerActions,offerRewards)
    
    if ((i_episode % TARGET_UPDATE) == 0) & (RANDOMPOLICY is False):
        for agent in world.agents:
            agent.updateTargetNets()

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

i=0
#Saving the episodic results
while os.path.isfile(fileName.format(i)):
    i += 1
a_file = open(fileName.format(i),"wb")
pickle.dump(argsDict,a_file)
a_file.close()

if PLOTTING is True:
    env.plotResult(argsDict)

env.close()
