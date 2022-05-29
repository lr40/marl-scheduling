# MARL Scheduling Environment

## Getting started

To train agents in the domain, run an arbitrary train*.py script in the src folder. You can customize the parameters explained below.

## Environment parameters

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
| dividedAgents             | Bool    | specifies if all the agents are of the distributed architecture. The difference in the name is due to a later name change.                            |
| aggregatedAgents          | Bool    | specifies if all the agents are of the semi-aggregated architecture                                                                                   |
| fullyAggregatedAgents     | Bool    | specifies if all the agents are of the fully aggregated architecture                                                                                  |
| locallySharedParameters   | Bool    | specifies if all the agents are of the distributed architecture with agent-wise parameter sharing                                                     |
| globallySharedParameters  | Bool    | specifies if all the agents in the domain are of the distributed architecture with global parameter sharing, i.e. all the agents share one neural net |

## PPO hyperparameters

| Parameter                             | type  | Description                                                                                                                             |
| --------------------------------------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| IS_PPO                                | Bool  | specifies if PPO is used                                                                                                                |
| LR_ACTOR                              | Float | initial learning rate of the actor network                                                                                              |
| LR_CRITIC                             | Float | initial learning rate of the critic network                                                                                             |
| ACCEPTOR_GAMMA                        | Float | the discount factor of future rewards for the acceptor unit                                                                             |
| OFFER_GAMMA                           | Float | the discount factor of future rewards for the offer unit                                                                                |
| RAW_K\_EPOCHS                         | Int   | used to determine the number of iterations with which the acceptor and offer units' memories are used to optimize their neural networks |
| ACCEPTOR_K_EPOCHS<br />OFFER_K_EPOCHS | Int   | not specified by the user but derived from RAW_K_EPOCHS                                                                                 |
| CENTRALISATION\_SAMPLE                | Int   | specifies, for parameter sharing, how many randomly selected subunit memories are included for a training run                           |
| EPS_CLIP                              | Float | specifies the value of the clipping parameter needed for PPO                                                                            |
| UPDATE\_STEP                          | Int   | specifies after how many time steps the neural networks are trained with the transitions experienced during this period                 |
| NUM_NEURONS                           | Int   | specifies the amount of neurons per hidden layer of one neural network                                                                  |

## Parameters of the experiments

The section 'The effect of intra-agent trading' uses the same hyperparameters as the section 'Agent architecture and scheduling performance'.

### Section: Agent architecture and scheduling performance

| Parameter               | 2 agents  | 4 agents  |
| ------------------------- | ----------- | :---------- |
| freePrices              | False     | False     |
| num_episodes            | 6000      | 6000      |
| episodeLength           | 100       | 100       |
| numberOfAgents          | 2         | 4         |
| numberOfCores           | 2         | 4         |
| newJobsPerRoundPerAgent | 1         | 1         |
| collectionLength        | 3         | 3         |
| possibleJobPriorities   | [3,10]    | [3,10]    |
| possibleJobLengths      | [6,3]     | [6,3]     |
| fixPricesList           | [2,7]     | [2,7]     |
| probabilities           | [0.8,0.2] | [0.8,0.2] |

| Parameter             | distributed | semi-aggregated | fully aggregated | distributed + local parameter sharing (2 agents) | distributed + local parameter sharing (4 agents) |
| ----------------------- | ------------- | ----------------- | ------------------ | -------------------------------------------------- | -------------------------------------------------- |
| LR_ACTOR              | 0.003       | 0.003           | 0.003            | 0.003                                            | 0.003                                            |
| LR_CRITIC             | 0.01        | 0.01            | 0.01             | 0.01                                             | 0.01                                             |
| ACCEPTOR_GAMMA        | 0.8733      | 0.8733          | 0.8733           | 0.8733                                           | 0.8733                                           |
| OFFER_GAMMA           | 0.5         | 0.5             | 0.5              | 0.5                                              | 0.5                                              |
| RAW_K_EPOCHS          | 3           | 3               | 3                | 3                                                | 3                                                |
| ACCEPTOR_K_EPOCHS     | 3           | 3               | 3                | 2                                                | 1                                                |
| OFFER_K_EPOCHS        | 3           | 3               | 3                | 1                                                | 1                                                |
| EPS_CLIP              | 0.2         | 0.2             | 0.2              | 0.2                                              | 0.2                                              |
| UPDATE_STEP           | 200         | 200             | 200              | 200                                              | 200                                              |
| NUM_NEURONS           | 16          | 32              | 64               | 16                                               | 16                                               |
| CENTRALISATION_SAMPLE | /           | /               | /                | 2                                                | 2                                                |

### Section: Price level and scarcity

| Parameter               | 2 cores | 4 cores |
| ------------------------- | --------- | :-------- |
| freePrices              | True    | True    |
| num_episodes            | 4000    | 4000    |
| episodeLength           | 100     | 100     |
| numberOfAgents          | 2       | 2       |
| numberOfCores           | 2       | 4       |
| newJobsPerRoundPerAgent | 1       | 1       |
| collectionLength        | 3       | 3       |
| possibleJobPriorities   | [5]     | [5]     |
| possibleJobLengths      | [5]     | [5]     |
| probabilities           | [1]     | [1]     |

| Parameter         | distributed with price setter network |
| ------------------- | --------------------------------------- |
| LR_ACTOR          | 0.003                                 |
| LR_CRITIC         | 0.01                                  |
| ACCEPTOR_GAMMA    | 0.95                                  |
| OFFER_GAMMA       | 0.5                                   |
| RAW_K_EPOCHS      | 2                                     |
| ACCEPTOR_K_EPOCHS | 2                                     |
| OFFER_K_EPOCHS    | 2                                     |
| EPS_CLIP          | 0.2                                   |
| UPDATE_STEP       | 200                                   |
| NUM_NEURONS       | 16                                    |

### Section: Price level and scheduling

| Parameter               | value               |
| ------------------------- | :-------------------- |
| freePrices              | True                |
| num_episodes            | 4000                |
| episodeLength           | 100                 |
| numberOfAgents          | 2                   |
| numberOfCores           | 3                   |
| newJobsPerRoundPerAgent | 1                   |
| collectionLength        | 3                   |
| possibleJobPriorities   | [2,4,8]             |
| possibleJobLengths      | [5,5,5]             |
| probabilities           | [(1/3),(1/3),(1/3)] |

| Parameter         | distributed with price setter network |
| ------------------- | --------------------------------------- |
| LR_ACTOR          | 0.003                                 |
| LR_CRITIC         | 0.01                                  |
| ACCEPTOR_GAMMA    | 0.95                                  |
| OFFER_GAMMA       | 0.5                                   |
| RAW_K_EPOCHS      | 2                                     |
| ACCEPTOR_K_EPOCHS | 2                                     |
| OFFER_K_EPOCHS    | 2                                     |
| EPS_CLIP          | 0.2                                   |
| UPDATE_STEP       | 200                                   |
| NUM_NEURONS       | 16                                    |

## Used code templates

The implementation of the reinforcement learning algorithms was based on freely available code templates, which should not go unmentioned: The implementation of the PPO algorithm was originally taken from [PPO](https://github.com/nikhilbarhate99/PPO-PyTorch "PPO-Pytorch implementation") and adapted to the requirements of the project. The implementation of the [DQN](https://github.com/pytorch/tutorials/blob/master/intermediate\_source/reinforcement\_q\_learning.py "DQN-Pytorch implementation") algorithm was also based on a public repository template and adapted to the project.

## Loop

The image below gives an overview about the process in the scheduling environment.

<img src="img/loop.png"/>
