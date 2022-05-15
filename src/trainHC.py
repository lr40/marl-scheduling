import statistics
from world import *
import SavingAndLoading
from SchedulingEnvironment import *


original_stdout = sys.__stdout__

#Setup of this run
PATH = 'C:\\Users\\lenna\\Desktop\\marl-scheduling_Neuanfang\\savedStateDicts\\savedDicts.pth'
outputFileName = 'Output.txt'
SAVING = False
LOADING = False
RENDERING = True


#Parameters of the world
RANDOMPOLICY = False
num_episodes = 50
episodeLength = 50
numberOfAgents = 2
numberOfCores = 3
possibleJobPriorities = [5]
possibleJobLengths = [4]
meanJobFraction = statistics.mean([F(a,b) for a,b in zip(possibleJobPriorities,possibleJobLengths)])
collectionLength = 2 
newJobsPerRoundPerAgent = 1
rewardMultiplier = 2 
assert (len(possibleJobLengths)==len(possibleJobPriorities))


world_params_dict = {'num_episodes': num_episodes, 'episodeLength': episodeLength, 'numberOfAgents': numberOfAgents, 'numberOfCores': numberOfCores,\
    'possibleJobPriorities': possibleJobPriorities, 'possibleJobLengths': possibleJobLengths,'collectionLength': collectionLength,\
    'newJobsPerRoundPerAgent': newJobsPerRoundPerAgent, 'rewardMultiplier': rewardMultiplier }
world = World(world_params_dict)
if LOADING:
    SavingAndLoading.loadCheckpoint(PATH,world)



RLparamsDict = {'RANDOMPOLICY': False, 'IS_HC': True}

env = SchedulingEnv(world,RLparamsDict)

parameters = dict(world_params_dict,**RLparamsDict)


averageEpisodicOfferRewards = []
averageEpisodicAceptorRewards = []

time1 = time.time()
#training loop
for i_episode in range(num_episodes):
    sys.stdout = original_stdout
    time2 = time.time()
    print(i_episode)
    print("Time: {}".format(time2 - time1))
    print("RandomPolicy: {}".format(RANDOMPOLICY))
    time1 = time.time()

    newAcceptorObservationTensors,newOfferObservationTensors, newAuctioneerObservation = env.reset()

    accumulatedOfferNetReward = torch.tensor([[[0] for _ in range(collectionLength)] for _ in range(numberOfAgents)])
    accumulatedAcceptorNetReward = torch.tensor([[[0] for _ in range(numberOfCores)] for _ in range(numberOfAgents)])
    for t in count():
        acceptorActions,offerActions \
            = env.getActionForAllDividedAgents(newAcceptorObservationTensors,newOfferObservationTensors)
        
        auctioneer_action = world.auctioneer.getAuctioneerAction(newAuctioneerObservation)
        
        #Die neuen Beobachtungen werden nach dem step() die alten sein.
        oldAcceptorObservationTensors = newAcceptorObservationTensors
        env.oldAcceptorObservationTensors = oldAcceptorObservationTensors
        oldOfferObservationTensors = newOfferObservationTensors
        env.oldOfferObservationTensors = oldOfferObservationTensors

        newAcceptorObservationTensors,newOfferObservationTensors, newAuctioneerObservation,\
        offerNetRewards, acceptorNetRewards, auctioneerReward, done = env.step(offerActions,acceptorActions,auctioneer_action)

        env.newAcceptorObservationTensors = newAcceptorObservationTensors
        env.newOfferObservationTensors = newOfferObservationTensors
        
        accumulatedOfferNetReward += offerNetRewards
        accumulatedAcceptorNetReward += acceptorNetRewards
        #Summiert zunächst in beiden Reward-Arten agentenweise auf und addiert dann je Agent die beiden Arten zusammen
        totalAgentRewards =[(i+j) for i,j in zip([sum(agentEntry.squeeze(1))for agentEntry in offerNetRewards],[sum(agentEntry.squeeze(1))for agentEntry in acceptorNetRewards])]
        
        if RENDERING:
            with open(outputFileName,'a') as output2:
                sys.stdout = output2
                env.render()
                pprint.pprint("offerNetRewards: {}".format(offerNetRewards.tolist()))
                pprint.pprint("acceptorNetRewards: {}".format(acceptorNetRewards.tolist()))
                pprint.pprint("auctioneerRewardPerCore: {}".format(auctioneerReward.tolist()))
                pprint.pprint("totalAgentRewards: {}".format(totalAgentRewards))
                print("RandomPolicy: {}".format(RANDOMPOLICY))
                sys.stdout = original_stdout
        
        if done:
            averageEpisodicOfferRewards.append((accumulatedOfferNetReward / episodeLength).numpy().mean())
            averageEpisodicAceptorRewards.append((accumulatedAcceptorNetReward / episodeLength).numpy().mean())
            break
        

if SAVING:
    SavingAndLoading.saveCheckpoint(PATH,world)

plotFixPricesResult(world,averageEpisodicAceptorRewards,averageEpisodicOfferRewards,meanJobFraction,parameters)
plt.show()

env.close()