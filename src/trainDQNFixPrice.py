"""
import sys
from world import *
from itertools import count
import SavingAndLoading
from SchedulingEnvironment import *
import statistics
"""
import pickle
import statistics
import sys
import time
from contextlib import redirect_stdout
from itertools import count

import numpy as np
import torch

from Plot import *
from PPOmodules import *
from SchedulingEnvironment import *
from world import *

sys.stdout = sys.__stdout__
original_stdout = sys.stdout

# Setup of this run
PATH = "C:\\Users\\lenna\\Desktop\\marl-scheduling_Neuanfang\\savedStateDicts\\savedDicts.pth"
outputFileName = "Output.txt"
SAVING = False
LOADING = False
RENDERING = False


# Parameters of the world
num_episodes = 100
episodeLength = 50
numberOfAgents = 2
numberOfCores = 3
rewardMultiplier = 2
possibleJobPriorities = [7, 5, 3]
possibleJobLengths = [5, 5, 5]
fixPricesList = [5, 3, 1]
probabilities = [0.3, 0.3, 0.4]
meanJobFraction = statistics.mean(
    [F(a, b) for a, b in zip(possibleJobPriorities, possibleJobLengths)]
)
collectionLength = 2
newJobsPerRoundPerAgent = 1
maxVisibleOffers = 4
assert len(possibleJobLengths) == len(possibleJobPriorities)
assert sum(probabilities) == 1

world_params_dict = {
    "num_episodes": num_episodes,
    "episodeLength": episodeLength,
    "numberOfAgents": numberOfAgents,
    "numberOfCores": numberOfCores,
    "possibleJobPriorities": possibleJobPriorities,
    "possibleJobLengths": possibleJobLengths,
    "collectionLength": collectionLength,
    "probabilities": probabilities,
    "newJobsPerRoundPerAgent": newJobsPerRoundPerAgent,
    "rewardMultiplier": rewardMultiplier,
    "freePrices": False,
    "fixPricesList": fixPricesList,
    "maxVisibleOffers": maxVisibleOffers,
}
world = World(world_params_dict)
if LOADING:
    SavingAndLoading.loadCheckpoint(PATH, world)

# RL Hyperparameters
IS_DQN = True
RANDOMPOLICY = False
BATCH_SIZE = 10
OFFER_GAMMA = 0.5
maxJobLength = max(possibleJobLengths)
ACCEPTOR_GAMMA = -((1 - maxJobLength) / maxJobLength)
ACCEPTOR_GAMMA = 0.84
RUN_START = 0.9
RUN_END = 0.05
RUN_DECAY = 500
REPLAY_MEMORY_SIZE = 10000
TARGET_UPDATE = 1
netZeroOfferReward = 0.5

RLparamsDict = {
    "BATCH_SIZE": BATCH_SIZE,
    "OFFER_GAMMA": OFFER_GAMMA,
    "ACCEPTOR_GAMMA": ACCEPTOR_GAMMA,
    "RUN_START": RUN_START,
    "REPLAY_MEMORY_SIZE": REPLAY_MEMORY_SIZE,
    "RUN_END": RUN_END,
    "RUN_DECAY": RUN_DECAY,
    "TARGET_UPDATE": TARGET_UPDATE,
    "RANDOMPOLICY": RANDOMPOLICY,
    "IS_DQN": IS_DQN,
    "freePrices": False,
    "netZeroOfferReward": netZeroOfferReward,
}

env = DQNDividedFixedPricesEnv(world, RLparamsDict)

parameters = dict(world_params_dict, **RLparamsDict)

averageEpisodicOfferRewards = []
averageEpisodicAcceptorRewards = []
averageEpisodicPrices = []
averageEpisodicAuctioneerReward = []
averageEpisodicDwellTimes = []

time1 = time.time()
# training loop
for i_episode in range(num_episodes):
    time2 = time.time()
    print(i_episode)
    print("Time: {}".format(time2 - time1))
    print("RandomPolicy: {}".format(RANDOMPOLICY))
    time1 = time.time()

    (
        newAcceptorObservationTensors,
        newOfferObservationTensors,
        newAuctioneerObservation,
    ) = env.reset()

    accumulatedOfferNetReward = torch.tensor(
        [[[0] for _ in range(collectionLength)] for _ in range(numberOfAgents)]
    )
    accumulatedAcceptorNetReward = torch.tensor(
        [[[0] for _ in range(numberOfCores)] for _ in range(numberOfAgents)]
    )
    prices = []
    collectedAuctioneerReward = []
    for t in count():
        acceptorActions, offerActions = env.getActionForAllAgents(
            newAcceptorObservationTensors, newOfferObservationTensors
        )

        auctioneer_action = world.auctioneer.getAuctioneerAction(newAuctioneerObservation)

        # Die neuen Beobachtungen werden nach dem step() die alten sein.
        oldAcceptorObservationTensors = newAcceptorObservationTensors
        env.oldAcceptorObservationTensors = oldAcceptorObservationTensors
        oldOfferObservationTensors = newOfferObservationTensors
        env.oldOfferObservationTensors = oldOfferObservationTensors

        (
            newAcceptorObservationTensors,
            newOfferObservationTensors,
            newAuctioneerObservation,
            offerNetRewards,
            acceptorNetRewards,
            auctioneerReward,
            agentReward,
            acceptionQuality,
            done,
        ) = env.step(offerActions, acceptorActions, auctioneer_action)

        for offer in world.acceptedOffers:
            price = offer.offeredReward
            prices.append((price, offer.jobKind))

        env.newAcceptorObservationTensors = newAcceptorObservationTensors
        env.newOfferObservationTensors = newOfferObservationTensors

        accumulatedOfferNetReward += offerNetRewards
        accumulatedAcceptorNetReward += acceptorNetRewards
        collectedAuctioneerReward.append(sum(auctioneerReward.tolist()))
        # Summiert zunächst in beiden Reward-Arten agentenweise auf und addiert dann je Agent die beiden Arten zusammen
        # totalAgentRewards =[(i+j) for i,j in zip([sum(agentEntry.squeeze(1))for agentEntry in offerNetRewards],[sum(agentEntry.squeeze(1))for agentEntry in acceptorNetRewards])]

        if RENDERING:
            with open(outputFileName, "a") as output2:
                sys.stdout = output2
                env.render()
                pprint.pprint("offerNetRewards: {}".format(offerNetRewards.tolist()))
                pprint.pprint("acceptorNetRewards: {}".format(acceptorNetRewards.tolist()))
                pprint.pprint("auctioneerRewardPerCore: {}".format(auctioneerReward.tolist()))
                # pprint.pprint("totalAgentRewards: {}".format(totalAgentRewards))
                print("RandomPolicy: {}".format(RANDOMPOLICY))
                sys.stdout = original_stdout

        if done:
            acc = []
            for i in range(len(possibleJobPriorities)):
                verweilzeiten = [
                    t.normalisierte_Verweilzeit
                    for t in world.verweilzeiten
                    if (t.Priorität == possibleJobPriorities[i])
                    & (t.Bedienzeit == possibleJobLengths[i])
                ]
                if verweilzeiten != []:
                    acc.append(statistics.mean(verweilzeiten))
                else:
                    acc.append(None)
            world.verweilzeiten = []
            averageEpisodicDwellTimes.append(acc)
            acc1 = []
            for i in range(len(possibleJobPriorities)):
                listComp = [tup[0] for tup in prices if tup[1] == i]
                if listComp != []:
                    acc1.append(statistics.mean(listComp))
                else:
                    acc1.append(None)
            averageEpisodicPrices.append(acc1)
            averageEpisodicOfferRewards.append(
                (accumulatedOfferNetReward / episodeLength).numpy().mean()
            )
            averageEpisodicAcceptorRewards.append(
                (accumulatedAcceptorNetReward / episodeLength).numpy().mean()
            )
            averageEpisodicAuctioneerReward.append(statistics.mean(collectedAuctioneerReward))
            break

        if RANDOMPOLICY is False:
            env.updateAcceptorMemoriesAndOptimize(acceptorActions, acceptorNetRewards)
            env.updateOfferMemoriesAndOptimize(offerActions, offerNetRewards)

    if ((i_episode % TARGET_UPDATE) == 0) & (RANDOMPOLICY is False):
        if env.isDQN is True:
            for agent in world.agents:
                agent.updateTargetNets()
        else:
            env.updateCentralTargetNets()


if SAVING:
    SavingAndLoading.saveCheckpoint(PATH, world)

plotFixPricesResult(
    world,
    averageEpisodicAcceptorRewards,
    averageEpisodicOfferRewards,
    averageEpisodicPrices,
    averageEpisodicAuctioneerReward,
    averageEpisodicDwellTimes,
    meanJobFraction,
    parameters,
)
plt.show()

env.close()
