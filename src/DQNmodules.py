import math, random, numpy as np
from collections import namedtuple, deque
import random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

Transition = namedtuple('Transition',
                        ('state','action','next_state','reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = np.array([None]*capacity,dtype=object)
        self.nextFreeIndex = 0
        self.capacity = capacity

    def push(self, *args):
        """Save a transition"""
        if (self.nextFreeIndex + 1) < self.capacity:
            self.memory[self.nextFreeIndex] = Transition(*args)
            self.nextFreeIndex += 1
        if (self.nextFreeIndex + 1) == self.capacity:
            self.memory[random.randint(0,(self.capacity-1))] = Transition(*args)

    def sample(self, batch_size):
        return np.random.choice(self.memory[:self.nextFreeIndex], batch_size)

    def __len__(self):
        return len(self.memory)

class DQNEntity(nn.Module):
    def __init__(self,world,env,amountInputChannels,numberOfActions):
        super(DQNEntity, self).__init__()
        self.world = world
        self.amountInputChannels = amountInputChannels
        self.numberOfActions = numberOfActions

        self.model = nn.Sequential(
                        nn.Linear(amountInputChannels, 16),
                        nn.Tanh(),
                        nn.Linear(16, numberOfActions),
                        #nn.ReLU()
                    )

        self.RUN_START = env.RUN_START
        self.RUN_END = env.RUN_END
        self.RUN_DECAY = env.RUN_DECAY

    def forward(self,x):
        x = x.float()
        return self.model(x)

    def selectAction(self,observationTensor):
        if self.world.randomPolicy:
            return self.generateRandomAction()

        sample = random.random()
        eps_treshold = self.RUN_END + (self.RUN_START - self.RUN_END)*math.exp(-1. * self.world.round / self.RUN_DECAY)

        if sample > eps_treshold:
            with torch.no_grad():
                # Hat also einfach das Format Ganzzahl ohne Tensor, ohne alles.
                return self.forward(observationTensor).max(0)[1].item()
        else:
            return self.generateRandomAction()
    
    def generateRandomAction(self):
        #Die Aktionen beziehen sich auf den Index der Angebote an diesen Agenten, an diesen Kern. 
        #Die Aktion numberOfAction ist länger als der Index steht für die Ablehnung.
        #0 für die Annahme des ersten Angebots (Index 0) usw.
        return float(random.randrange(self.numberOfActions))

class DQNAcceptorNet(DQNEntity):
    def __init__(self, world,env):
        # Beobachtung der Priorität und verbleibenden Länge auf einem Kern (plus Ownership-Flag), sowie angebotenem Reward und Länge je Angebot
        amountInputChannels = 3 + (2*world.maxAmountOfOffersToOneAgent)
        # Annahme eines Angebots oder gar keines Angebots
        numberOfActions = world.maxAmountOfOffersToOneAgent + 1
        super(DQNAcceptorNet, self).__init__(world,env,amountInputChannels,numberOfActions)
        
class DQNOfferNet(DQNEntity):
    def __init__(self,world,env):
        # Für jeden Kern werden der Brutto-Reward (Priorität) und die verbleibende Zeit beobachtet und für den eigenen QueueSlot auch.
        amountInputChannels = (world.numberOfCores * 2) + 2
        #Für diesen Queue-Slot kann also letztlich ein Kern ausgewählt werden oder an gar keinen Kern ein Angebot gemacht werden (+1).
        numberOfActions = (world.numberOfCores) + 1 
        super(DQNOfferNet, self).__init__(world,env,amountInputChannels,numberOfActions)
        


def optimize_model(env,memory,BATCH_SIZE,policy_net,target_net,GAMMA,optimizer):
        #reward tensor([[0]]), oldObsTensor([[-1,-1,...,-1]]), newObsTensor([[-1,-1,...,-1]]), actionTensor([1])
        if len(memory) < BATCH_SIZE:
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        transitions = memory.sample(BATCH_SIZE)

        batch = Transition(*zip(*transitions))
        #Für batch_size = 2
        #Hat's das Format Transition(state=((1,1...,1),(2,2...,2)), action=(1,2), next_state=((n1,n1,...,n1),(n2,n2,...,n2)), reward=(1,2))

        non_final_mask = torch.tensor (tuple(map(lambda s: s is not None, batch.next_state)), device = "cpu", dtype=torch.bool)
        #Hat das Format torch.tensor([bool,bool,...])

        non_final_next_states = torch.cat([torch.tensor(s).unsqueeze(0) for s in batch.next_state if s is not None])
        

        state_batch = torch.cat([torch.tensor(state).unsqueeze(0) for state in batch.state])
        action_batch = torch.cat([torch.tensor(action).unsqueeze(0).unsqueeze(0) for action in batch.action])
        reward_batch = torch.cat([torch.tensor(reward).unsqueeze(0) for reward in batch.reward])

        #Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        #columns of actions taken. These are the actions (nicht action-values?) which would've been taken
        #for each batch state according to the policy_net (nicht eher so, dass sie auch zufällig gewählt worden sein können?)
        state_action_values = policy_net(state_batch).gather(1, action_batch.long())

        #Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.  - Schlau!
        next_state_values = torch.zeros(BATCH_SIZE, device = "cpu")
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

        #Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch.squeeze(1)

        #hier tritt eine Nutzer-Warnung auf: ungleiche Formate der beiden Argumente
        # Vermutung: Könnte es einen Unterschied in den Formaten bei acceptor- und offerNets geben? Tritt die Warnung nur in bestimmten Konstellationen auf?
        # Morgen hier weiter untersuchen und sicherstellen, dass alles ohne Probleme läuft! 

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values,expected_state_action_values.unsqueeze(1))

        #Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1,1)
            #Glättet die Gradienten also ein auf das Intervall [-1,1]
        optimizer.step()

    



    







