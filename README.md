# MARL Scheduling Environment

## Getting started

To train agents in the domain, run an arbitrary train*.py script in the src folder. You can customize the parameters explained below.

## Scenario parameters


| Parameter                 | type    | Description                                                                                                                                           |
| :-------------------------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| numberOfCores             | Int     | amount of compute cores in the domain                                                                                                                 |
| numberOfAgents            | Int     | amount of RL agents in the domain                                                                                                                     |
| collectionLength          | Int     | amount of associated job slots of each agent                                                                                                          |
| possibleJobPriorities     | [Int]   | job priority for each job type. list index is the job type                                                                                            |
| possibleJobLengths        | [Int]   | job length for each job type. list index is the job type                                                                                              |
| fixPricesList             | [Int]   | fix price for each job type. list index is the job type                                                                                               |
| probabilities             | [Float] | spawn probability for each job type. list index is the job type. Its sum must equal 1.                                                                |
| newJobsPerRoundPerAgent   | Int     | if one agent has capacity for new jobs in his job collection: how many new jobs are generated at the beginning of each round                          |
| freePrices                | Bool    | specifies if free prices are used                                                                                                                     |
| commercialFreePriceReward | Bool    | specifies in case of free prices if the commercial or non-commercial reward function is used                                                          |
| dividedAgents             | Bool    | specifies if all the agents are of the distributed architecture                                                                                       |
| aggregatedAgents          | Bool    | specifies if all the agents are of the semi-aggregated architecture                                                                                   |
| fullyAggregatedAgents     | Bool    | specifies if all the agents are of the fully aggregated architecture                                                                                  |
| locallySharedParameters   | Bool    | specifies if all the agents are of the distributed architecture with agent-wise parameter sharing                                                     |
| globallySharedParameters  | Bool    | specifies if all the agents in the domain are of the distributed architecture with global parameter sharing, i.e. all the agents share one neural net |


## PPO hyperparameters


| Parameter              | type  | Description                                                                                                                             |
| ------------------------ | ------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| IS_PPO                 | Bool  | specifies if PPO is used                                                                                                                |
| LR_ACTOR               | Float | initial learning rate of the actor network                                                                                              |
| LR_CRITIC              | Float | initial learning rate of the critic network                                                                                             |
| ACCEPTOR_GAMMA         | Float | the discount factor of future rewards for the acceptor unit                                                                             |
| OFFER_GAMMA            | Float | the discount factor of future rewards for the offer unit                                                                                |
| RAW_K\_EPOCHS          | Int   | used to determine the number of iterations with which the acceptor and offer units' memories are used to optimize their neural networks |
| CENTRALISATION\_SAMPLE | Int   | specifies, for parameter sharing, how many randomly selected subunit memories are included for a training run                           |
| EPS_CLIP               | Float | specifies the value of the clipping parameter needed for PPO                                                                            |
| UPDATE\_STEP           | Int   | specifies after how many time steps the neural networks are trained with the transitions experienced during this period                 |


## Used code templates

The implementation of the reinforcement learning algorithms was based on freely available code templates, which should not go unmentioned: The implementation of the PPO algorithm was originally taken from [PPO](https://github.com/nikhilbarhate99/PPO-PyTorch "PPO-Pytorch implementation") and adapted to the requirements of the project. The implementation of the [DQN](https://github.com/pytorch/tutorials/blob/master/intermediate\_source/reinforcement\_q\_learning.py "DQN-Pytorch implementation") algorithm was also based on a public repository template and adapted to the project.

## Loop

The image below gives an overview about the process in the scheduling environment.

<img src="img/loop.png"/>
