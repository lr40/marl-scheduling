import copy
import pprint
import statistics
import time

import gym
import numpy as np
import torch
import torch.optim as optim
from gym import error, spaces, utils

import SavingAndLoading
from Agent import *
from Auctioneer import *
from DQNmodules import *
from Plot import *
from Reward import *
from world import *


class SchedulingEnv(gym.Env):
    def __init__(self, world, params):
        self.world = world
        self.formerCorePrios = [-1 for _ in range(world.numberOfCores)]
        self.formerCoreLengths = [-1 for _ in range(world.numberOfCores)]
        self.correspondingOfferIDs = [
            [[] for _ in range(world.numberOfCores)] for _ in range(world.numberOfAgents)
        ]
        self.auctioneer_correspondingOfferIDs = []
        self.netZeroOfferReward = params["netZeroOfferReward"]

    def step(self, offerActions, acceptorActions, auctioneer_action):
        self.world.step1(
            offerActions,
            acceptorActions,
            self.correspondingOfferIDs,
            auctioneer_action,
            self.auctioneer_correspondingOfferIDs,
        )

        allAcceptorObservationTensors = []
        allOfferObservationTensors = []

        for agent in self.world.agents:
            (
                acceptorNetObservationTensors,
                acceptorNetOfferIDs,
                offerNetObservationTensors,
            ) = agent.gatherObservations()
            allAcceptorObservationTensors.append(acceptorNetObservationTensors)
            self.correspondingOfferIDs[agent.agentID - 1] = acceptorNetOfferIDs
            allOfferObservationTensors.append(offerNetObservationTensors)

        (
            auctioneer_observation,
            auctioneer_correspondingOfferIDs,
        ) = gatherDividedAuctioneerObservation(self.world)
        self.auctioneer_correspondingOfferIDs = auctioneer_correspondingOfferIDs

        acceptionQualityMean = self.calculateAverageAcceptionQuality()

        offerRewards, acceptorRewards, auctioneerReward, agentReward = self.getRewards()

        if (self.world.round % self.world.episodeLength) == 0:
            done = True
        else:
            done = False

        # Aktualisiere die vergangenen Kern-Prioritäten, die für die zukünftige, korrekte Reward-Ermittlung der Angebots-netze benötigt werden.
        self.formerCorePrios = [core.job.priority for core in self.world.cores]
        self.formerCoreLengths = [core.job.remainingLength for core in self.world.cores]

        return (
            allAcceptorObservationTensors,
            allOfferObservationTensors,
            auctioneer_observation,
            offerRewards,
            acceptorRewards,
            auctioneerReward,
            agentReward,
            acceptionQualityMean,
            done,
        )

    def reset(self):
        allAcceptorNetObservationTensors = []
        allOfferNetObservationTensors = []

        for agent in self.world.agents:
            (
                acceptorNetObservationTensors,
                acceptorNetOfferIDs,
                offerNetObservationTensors,
            ) = agent.gatherObservations()
            allAcceptorNetObservationTensors.append(acceptorNetObservationTensors)
            self.correspondingOfferIDs[agent.agentID - 1] = acceptorNetOfferIDs
            allOfferNetObservationTensors.append(offerNetObservationTensors)

        (
            auctioneer_observation,
            auctioneer_correspondingOfferIDs,
        ) = gatherDividedAuctioneerObservation(self.world)
        self.auctioneer_correspondingOfferIDs = auctioneer_correspondingOfferIDs

        return (
            allAcceptorNetObservationTensors,
            allOfferNetObservationTensors,
            auctioneer_observation,
        )

    def render(self, mode="human"):
        print("___________________________________")
        pprint.pprint("Round: ")
        print(self.world.round)
        if mode == "human":
            print("Core-States:")
            for core in self.world.cores:
                pprint.pprint(core.getCoreState())
            pprint.pprint(" ")
            print("Agent-States:")
            for agent in self.world.agents:
                pprint.pprint(agent.getAgentState())
        pprint.pprint(" ")
        print("Auctioneer-State:")
        pprint.pprint(self.world.auctioneer.getAuctioneerState())
        pprint.pprint(" ")
        print("Offer-States:")
        for offer in self.world.offers:
            pprint.pprint(offer.getOfferState())
        print("Job-Termination-Info:")
        for core, ownerID, jobID, generatedReward, round1 in self.world.jobTerminationInfo:
            print(
                "coreID: {}, ownerID: {}, jobID: {}, generatedReward: {}, round: {}".format(
                    core.coreID, ownerID, jobID, generatedReward, round1
                )
            )
        print("Accepted Offers:")
        for offer in self.world.acceptedOffers:
            print(
                "coreID: {}, offererID: {}, recipientID: {}, jobID: {}, offeredReward {}, slot: {}".format(
                    offer.coreID,
                    offer.offererID,
                    offer.recipientID,
                    offer.jobID,
                    offer.offeredReward,
                    offer.queuePosition,
                )
            )

    def getActionForAllAgents(
        self, nestedAcceptorNetObservationTensors, nestedOfferNetObservationTensors
    ):
        nestedAcceptorNetActionTensors = list(
            [
                [-5 for core in range(self.world.numberOfCores)]
                for agent in range(self.world.numberOfAgents)
            ]
        )
        nestedOfferNetActionTensors = list(
            [
                [-5 for slot in range(self.world.collectionLength)]
                for agent in range(self.world.numberOfAgents)
            ]
        )
        for i, agent in enumerate(self.world.agents):
            offerNetActions, acceptorNetActions = agent.getActions(
                nestedOfferNetObservationTensors[i], nestedAcceptorNetObservationTensors[i]
            )
            nestedAcceptorNetActionTensors[i] = acceptorNetActions
            nestedOfferNetActionTensors[i] = offerNetActions

        return nestedAcceptorNetActionTensors, nestedOfferNetActionTensors

    def calculateAverageAcceptionQuality(self):
        # bezieht sich nur auf Nicht-Auktionatoren, da nur sie lernen
        acceptionQualities = []
        nonAuctioneerOffers = [
            offer for offer in self.world.acceptedOffers if offer.recipientID != 0
        ]
        for offer in nonAuctioneerOffers:
            acceptionQuality = (offer.offeredReward / offer.necessaryTime) - (
                (self.formerCorePrios[offer.coreID - 1] / self.formerCoreLengths[offer.coreID - 1])
                if (self.formerCorePrios[offer.coreID - 1] != -1)
                else 0
            )
            acceptionQuality *= 10
            acceptionQualities.append(acceptionQuality)

        result = statistics.mean(acceptionQualities) if (acceptionQualities != []) else None
        amount = len(acceptionQualities)

        return result, amount


class PPOSchedulingEnv(SchedulingEnv):
    def __init__(self, world, params):
        super().__init__(world, params)
        self.LR_ACTOR = params["LR_ACTOR"]
        self.LR_CRITIC = params["LR_CRITIC"]
        self.OFFER_GAMMA = params["OFFER_GAMMA"]
        self.ACCEPTOR_GAMMA = params["ACCEPTOR_GAMMA"]
        self.EPS_CLIP = params["EPS_CLIP"]
        self.RAW_K_EPOCHS = params["RAW_K_EPOCHS"]
        self.ACCEPTOR_K_EPOCHS = params["ACCEPTOR_K_EPOCHS"]
        self.OFFER_K_EPOCHS = params["OFFER_K_EPOCHS"]
        self.CENTRALISATION_SAMPLE = params["CENTRALISATION_SAMPLE"]

    def updateAgents(self):
        for agent in self.world.agents:
            agent.updateParts()


class PPOAggregatedFixPriceEnv(PPOSchedulingEnv):
    def __init__(self, world, params):
        super().__init__(world, params)
        self.world.agents = [
            AggregatedFixPricePPOAgent(self.world, self) for _ in range(self.world.numberOfAgents)
        ]

    def getRewards(self):
        return getAggregatedFixedPricesReward(self)

    def saveRewards(self, offerUnitRewards, acceptorUnitRewards, agentReward):
        for i, agent in enumerate(self.world.agents):
            agent.saveRewards(offerUnitRewards[i], agentReward[i])

    def plotResult(self, argsDict):
        plotFixPricesResult(argsDict)


class PPOFullyAggregatedFixPriceEnv(PPOSchedulingEnv):
    def __init__(self, world, params):
        super().__init__(world, params)
        self.world.agents = [
            FullyAggregatedFixPricePPOAgent(self.world, self)
            for _ in range(self.world.numberOfAgents)
        ]

    def getRewards(self):
        return getAggregatedFixedPricesReward(self)

    # Ein bisschen verwirrend: agentReward meint hier die Rewardchain-Rewards
    def saveRewards(self, offerUnitRewards, acceptorUnitRewards, agentReward):
        # arbeitet einfach mit den halb-aggregierten Rewards, die dann auf der Agenten-Ebene addiert werden.
        for i, agent in enumerate(self.world.agents):
            fullyAggregatedReward = agentReward[i] + offerUnitRewards[i][0]
            agent.saveRewards(fullyAggregatedReward)  # offerUnitRewards[i],acceptorUnitRewards[i]

    def plotResult(self, argsDict):
        plotFixPricesResult(argsDict)


class PPODividedFreePriceEnv(PPOSchedulingEnv):
    def __init__(self, world, params, commercialFreePriceReward):
        super().__init__(world, params)
        self.world.agents = [
            DividedFreePricePPOAgent(self.world, self) for _ in range(self.world.numberOfAgents)
        ]
        self.commercialFreePriceReward = commercialFreePriceReward

    def getRewards(self):
        return getDividedFreePricesReward(self, self.commercialFreePriceReward)

    def saveRewards(self, offerNetRewards, acceptorNetRewards, agentReward):
        coreChooserRewards = offerNetRewards[0]
        priceChooserRewards = offerNetRewards[1]
        for i, agent in enumerate(self.world.agents):
            agent.saveRewards(coreChooserRewards[i], priceChooserRewards[i], acceptorNetRewards[i])

    def plotResult(self, argsDict):
        plotFreePricesResult(argsDict)


class PPODividedFixedPriceEnv(PPOSchedulingEnv):
    def __init__(self, world, params):
        super().__init__(world, params)
        self.world.agents = [
            DividedFixedPricePPOAgent(self.world, self) for _ in range(self.world.numberOfAgents)
        ]
        self.tradeRevenues = None
        self.termnationRevenues = None

    def getRewards(self):
        return getDividedFixedPricesReward(self)

    def saveRewards(self, offerNetRewards, acceptorNetRewards, agentReward):
        for i, agent in enumerate(self.world.agents):
            agent.saveRewards(offerNetRewards[i], acceptorNetRewards[i])

    def plotResult(self, argsDict):
        plotFixPricesResult(argsDict)


class GloballySharedParamsDividedFixedPriceEnv(PPOSchedulingEnv):
    def __init__(self, world, params):
        super().__init__(world, params)
        self.sharedAcceptorNet = GloballySharedAcceptorPPO(self.world, self)
        self.sharedOfferNet = GloballySharedOfferPPO(self.world, self)
        self.world.agents = [
            DividedFixedPriceGloballySharedPPOAgent(self.world, self)
            for _ in range(self.world.numberOfAgents)
        ]

    def saveRewards(self, offerNetRewards, acceptorNetRewards, agentReward):
        for i, agent in enumerate(self.world.agents):
            agent.saveRewards(offerNetRewards[i], acceptorNetRewards[i])

    def getRewards(self):
        return getDividedFixedPricesReward(self)

    def plotResult(self, argsDict):
        plotFixPricesResult(argsDict)

    def updateAgents(self):
        for i in range(self.CENTRALISATION_SAMPLE):
            lastUpdate = False
            randomIndex0 = random.randint(0, self.world.numberOfAgents - 1)
            randomIndex1 = random.randint(0, self.world.numberOfCores - 1)
            if i == (self.CENTRALISATION_SAMPLE - 1):
                lastUpdate = True
            self.sharedAcceptorNet.update(randomIndex0, randomIndex1, lastUpdate)

        for j in range(self.CENTRALISATION_SAMPLE):
            lastUpdate = False
            randomIndex0 = random.randint(0, self.world.numberOfAgents - 1)
            randomIndex1 = random.randint(0, self.world.collectionLength - 1)
            if j == (self.CENTRALISATION_SAMPLE - 1):
                lastUpdate = True
            self.sharedOfferNet.update(randomIndex0, randomIndex1, lastUpdate)


class LocallySharedParamsDividedFixedPriceEnv(PPOSchedulingEnv):
    def __init__(self, world, params):
        super().__init__(world, params)
        self.world.agents = [
            DividedFixedPriceLocallySharedPPOAgent(self.world, self)
            for _ in range(self.world.numberOfAgents)
        ]

    def saveRewards(self, offerNetRewards, acceptorNetRewards, agentReward):
        for i, agent in enumerate(self.world.agents):
            agent.saveRewards(offerNetRewards[i], acceptorNetRewards[i])

    def getRewards(self):
        return getDividedFixedPricesReward(self)

    def plotResult(self, argsDict):
        plotFixPricesResult(argsDict)


class DQNSchedulingEnv(SchedulingEnv):
    def __init__(self, world, params):
        super().__init__(world, params)
        self.oldOfferObservationTensors = []
        self.newOfferObservationTensors = []
        self.oldAcceptorObservationTensors = []
        self.newAcceptorObservationTensors = []
        self.RUN_END = params["RUN_END"]
        self.RUN_START = params["RUN_START"]
        self.RUN_DECAY = params["RUN_DECAY"]
        self.BATCH_SIZE = params["BATCH_SIZE"]
        self.OFFER_GAMMA = params["OFFER_GAMMA"]
        self.ACCEPTOR_GAMMA = params["ACCEPTOR_GAMMA"]
        self.REPLAY_MEMORY_SIZE = params["REPLAY_MEMORY_SIZE"]

    def updateOfferMemoriesAndOptimize(self, offerActions, offerNetRewards):
        for i, agent in enumerate(self.world.agents):
            oldObservations = self.oldOfferObservationTensors[i]
            newObservations = self.newOfferObservationTensors[i]
            offerMemories = agent.memories["offerMemories"]
            offerNetActions = offerActions[i]
            rewards = offerNetRewards[i]
            offerNets = agent.policy["OfferNets"]
            targetNets = agent.target["OfferNets"]
            optimizers = agent.optimizers["offerOptimizers"]
            for j in range(self.world.collectionLength):
                memory = offerMemories[j]
                action = offerNetActions[j]
                reward = rewards[j]
                oldObservation = tuple(oldObservations[j].tolist())
                newObservation = tuple(newObservations[j].tolist())
                memory.push(oldObservation, action, newObservation, reward)
                policyNet = offerNets[j]
                targetNet = targetNets[j]
                optimizer = optimizers[j]
                # Eventuell muss die Optimierung nicht immer durchgeführt werden, sondern nur periodisch.
                optimize_model(
                    self,
                    memory,
                    self.BATCH_SIZE,
                    policyNet,
                    targetNet,
                    self.OFFER_GAMMA,
                    optimizer,
                )

    def updateAcceptorMemoriesAndOptimize(self, acceptorActions, acceptorNetRewards):
        for i, agent in enumerate(self.world.agents):
            oldObservations = self.oldAcceptorObservationTensors[i]
            newObservations = self.newAcceptorObservationTensors[i]
            acceptorMemories = agent.memories["acceptorMemories"]
            acceptorNetActions = acceptorActions[i]
            rewards = acceptorNetRewards[i]
            acceptorNets = agent.policy["AcceptorNets"]
            targetNets = agent.target["AcceptorNets"]
            optimizers = agent.optimizers["acceptorOptimizers"]
            for j in range(self.world.numberOfCores):
                memory = acceptorMemories[j]
                action = acceptorNetActions[j]
                reward = rewards[j]
                oldObservation = tuple(oldObservations[j].tolist())
                newObservation = tuple(newObservations[j].tolist())
                memory.push(oldObservation, action, newObservation, reward)
                policyNet = acceptorNets[j]
                targetNet = targetNets[j]
                optimizer = optimizers[j]
                optimize_model(
                    self,
                    memory,
                    self.BATCH_SIZE,
                    policyNet,
                    targetNet,
                    self.ACCEPTOR_GAMMA,
                    optimizer,
                )


class DQNDividedFixedPricesEnv(DQNSchedulingEnv):
    def __init__(self, world, params):
        super().__init__(world, params)
        self.world.agents = [
            DividedFixPriceDQNAgent(self.world, self) for _ in range(self.world.numberOfAgents)
        ]

    def getRewards(self):
        return getDividedFixedPricesReward(self)


class HardcodedFixPriceEnvironment(SchedulingEnv):
    def __init__(self, world, params):
        super().__init__(world, params)
        self.world.agents = [
            DividedHardcodedAgent(self.world) for _ in range(self.world.numberOfAgents)
        ]

    def saveRewards(self, offerNetRewards, acceptorNetRewards, agentReward):
        ...

    def updateAgents(self):
        ...

    def getRewards(self):
        return getDividedFixedPricesReward(self)

    def plotResult(self, argsDict):
        plotFixPricesResult(argsDict)
