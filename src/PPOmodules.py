import torch
import torch.nn as nn
from torch.distributions import Categorical
from collections import deque
import math
class ExperienceBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        #self.is_terminals = []
    

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        #del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, amountInputChannels, numberOfActions, numberOfNeurons):
        super(ActorCritic, self).__init__()

        #gibt eine Wahrscheinlichkeits-Verteilung aus, eine Handlung zu wählen.
        #0 steht für die Annahme des ersten Angebots oder das Bieten auf den ersten Kern.
        #Die Handlung numberOfActions steht für die Ablehung eines jeden Angebots, oder dass kein Angebot gemacht wird.
        self.actor = nn.Sequential(
                        nn.Linear(amountInputChannels, numberOfNeurons),
                        nn.Tanh(),
                        nn.Linear(numberOfNeurons, numberOfNeurons),
                        nn.Tanh(),
                        nn.Linear(numberOfNeurons, numberOfActions),
                        nn.Softmax(dim=-1)
                    )

        #bewertet die Zustände
        self.critic = nn.Sequential(
                        nn.Linear(amountInputChannels, numberOfNeurons),
                        nn.Tanh(),
                        nn.Linear(numberOfNeurons, numberOfNeurons),
                        nn.Tanh(),
                        nn.Linear(numberOfNeurons, 1)
                    )
        
    def forward(self):
        raise NotImplementedError
    

    def act(self, state):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        #berücksichtigt also die Wahrscheinlichkeiten beim Ziehen.
        action = dist.sample()
        #Generell ist das Format: Die erste Handlung liefert 0 zurück, die zweite 1 usw. Wenn die Handlungen Indizes sein sollen, ist das wichtig zu wissen.
        # Einfach der ln() der Wahrscheinlichkeit.
        action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach()
    

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state).squeeze()
        
        return action_logprobs, state_values, dist_entropy

class PPO:
    def __init__(self, world, amountInputChannels,numberOfActions, lr_actor, lr_critic, gamma, eps_clip, K_epochs, numberOfNeurons):

        self.world = world
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.amountInputChannels = amountInputChannels
        self.numberOfActions = numberOfActions
        
        self.buffer = ExperienceBuffer()

        
        self.policy = ActorCritic(self.amountInputChannels, self.numberOfActions, numberOfNeurons)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(self.amountInputChannels, self.numberOfActions, numberOfNeurons)
        self.policy_old.load_state_dict(self.policy.state_dict())
        

        self.MseLoss = nn.MSELoss()


    def selectAction(self, state):

        with torch.no_grad():
            state = state.float()
            action, action_logprob = self.policy_old.act(state)
        
        #state und action werden hier also schon auf dieser Ebene abgespeichert.
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)

        return action.item()


    def update(self):
        # Monte Carlo estimate of returns
        rewards = deque([])
        discounted_reward = 0
        for reward in reversed(self.buffer.rewards):
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.appendleft(discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach()
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach()
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach()

        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

class AggregatedAcceptorPPO(PPO):
    def __init__(self,world,env):
        amountInputChannels = (3 + (2*world.maxAmountOfOffersToOneAgent))*world.numberOfCores
        numberOfActions = (world.maxAmountOfOffersToOneAgent + 1)**world.numberOfCores
        numberOfNeurons = 32
        super().__init__(world,amountInputChannels,numberOfActions,env.LR_ACTOR, env.LR_CRITIC, env.ACCEPTOR_GAMMA, env.EPS_CLIP,env.ACCEPTOR_K_EPOCHS,numberOfNeurons)

class AggregatedOfferPPO(PPO):
    def __init__(self,world,env):
        amountInputChannels = (world.numberOfCores * 2) + (world.collectionLength * 2)
        numberOfActions = (world.numberOfCores + 1)**world.collectionLength
        numberOfNeurons = 32
        super().__init__(world,amountInputChannels,numberOfActions,env.LR_ACTOR, env.LR_CRITIC, env.OFFER_GAMMA, env.EPS_CLIP,env.OFFER_K_EPOCHS,numberOfNeurons)

class FullyAggregatedPPO(PPO):
    def __init__(self,world,env):
        amountInputChannels = ((3 + (2*world.maxAmountOfOffersToOneAgent))*world.numberOfCores) + ((world.numberOfCores * 2) + (world.collectionLength * 2))
        numberOfActions = ((world.maxAmountOfOffersToOneAgent + 1)**world.numberOfCores)*((world.numberOfCores + 1)**world.collectionLength)
        numberOfNeurons = 64
        super().__init__(world,amountInputChannels,numberOfActions,env.LR_ACTOR, env.LR_CRITIC, env.ACCEPTOR_GAMMA, env.EPS_CLIP,env.ACCEPTOR_K_EPOCHS,numberOfNeurons)

class AcceptorPPO(PPO):
    def __init__(self,world,env):
        amountInputChannels = (3 + (2*world.maxAmountOfOffersToOneAgent) ) #+ (2*world.maxVisibleOffers)
        numberOfActions = (world.maxAmountOfOffersToOneAgent + 1)
        numberOfNeurons = 16
        super().__init__(world,amountInputChannels,numberOfActions,env.LR_ACTOR, env.LR_CRITIC, env.ACCEPTOR_GAMMA, env.EPS_CLIP,env.ACCEPTOR_K_EPOCHS,numberOfNeurons)

class OfferPPO(PPO):
    def __init__(self,world,env):
        amountInputChannels = ((world.numberOfCores * 2) + 2)
        numberOfActions = ((world.numberOfCores) + 1)
        numberOfNeurons = 16
        super().__init__(world,amountInputChannels,numberOfActions,env.LR_ACTOR, env.LR_CRITIC, env.OFFER_GAMMA, env.EPS_CLIP,env.OFFER_K_EPOCHS,numberOfNeurons)

class FreePriceOfferPPO(object):
    def __init__(self,world,env):
        self.world = world
        numberOfNeurons = 16
        self.coreChooserInputAmount = ((world.numberOfCores * 2) + 2)
        self.coreChoserNumOfActions = (world.numberOfCores + 1) # Die Möglichkeit, ein Angebot nicht zu machen, gibt es.
        self.coreChooser = PPO(world,self.coreChooserInputAmount,self.coreChoserNumOfActions,env.LR_ACTOR, env.LR_CRITIC, env.OFFER_GAMMA, env.EPS_CLIP,env.RAW_K_EPOCHS,numberOfNeurons)
        self.priceChooserInputAmount = 4 # prio, Länge für den gewählten Kern; prio, Länge für den eigenen Queue-Slot
        self.priceChoserNumOfActions = world.maxSumToOffer + 1
        self.priceChooser = PPO(world,self.priceChooserInputAmount,self.priceChoserNumOfActions,env.LR_ACTOR, env.LR_CRITIC, env.OFFER_GAMMA, env.EPS_CLIP,env.RAW_K_EPOCHS,numberOfNeurons)
    
    def update(self):
        self.coreChooser.update()
        self.priceChooser.update()
    
    def selectAction(self,coreChooserInput):
        coreChooserAction = self.coreChooser.selectAction(coreChooserInput)
        
        #Für die Möglichkeit, dass Kern-ID 0 ausgewählt wurde (kein Angebot), sollte der Price-Chooser auch nix wählen.
        if coreChooserAction != 0:
            priceChooserInput = torch.cat((coreChooserInput[(coreChooserAction*2):(coreChooserAction*2+2)],coreChooserInput[-2:]))
            priceChooserAction = self.priceChooser.selectAction(priceChooserInput)
        else:
            #Passiert nur, damit hinterher nicht mehr Rewards als Beobachtungen + Actions abgespeichert sind.
            #Andere Lösungen wären viel zu aufwendig.
            self.priceChooser.selectAction(torch.tensor([-5,-5,-5,-5]))
            priceChooserAction = -5   #Beliebiger, erkennbarer Wert, ums beim Debuggen einfacher erkennen zu können
        
        return coreChooserAction, priceChooserAction

class GloballySharedPPO:
    def __init__(self, world,subLen, amountInputChannels,numberOfActions, lr_actor, lr_critic, gamma, eps_clip, K_epochs,numberOfNeurons):

        self.world = world
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.amountInputChannels = amountInputChannels
        self.numberOfActions = numberOfActions
        
        self.buffers = [[ExperienceBuffer() for _ in range(subLen)] for _ in range(self.world.numberOfAgents)]
        self.subLen = subLen
        self.maxSubIndex = subLen - 1

        
        self.policy = ActorCritic(self.amountInputChannels, self.numberOfActions,numberOfNeurons)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(self.amountInputChannels, self.numberOfActions,numberOfNeurons)
        self.policy_old.load_state_dict(self.policy.state_dict())
        

        self.MseLoss = nn.MSELoss()


    def selectAction(self,agentID,subID, state):

        with torch.no_grad():
            state = state.float()
            action, action_logprob = self.policy_old.act(state)
        
        #state und action werden hier also schon auf dieser Ebene abgespeichert.
        self.buffers[agentID - 1][subID].states.append(state)
        self.buffers[agentID - 1][subID].actions.append(action)
        self.buffers[agentID - 1][subID].logprobs.append(action_logprob)

        return action.item()
    
    def updateOldPolicy(self):
        self.policy_old.load_state_dict(self.policy.state_dict())

    def update(self,randomIndex0,randomIndex1,lastUpdate):
        # Monte Carlo estimate of returns
        rewards = deque([])
        discounted_reward = 0
        for reward in reversed(self.buffers[randomIndex0][randomIndex1].rewards):
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.appendleft(discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffers[randomIndex0][randomIndex1].states, dim=0)).detach()
        old_actions = torch.squeeze(torch.stack(self.buffers[randomIndex0 ][randomIndex1].actions, dim=0)).detach()
        old_logprobs = torch.squeeze(torch.stack(self.buffers[randomIndex0][randomIndex1].logprobs, dim=0)).detach()

        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        #Beim letzten Agenten und der letzten Sub-Einheit wird die alte Policy aktualisiert
        if lastUpdate is True:
            for index0 in range(self.world.numberOfAgents):
                for index1 in range(self.subLen):
                    self.buffers[index0][index1].clear()
            self.updateOldPolicy()

class GloballySharedAcceptorPPO(GloballySharedPPO):
    def __init__(self,world,env):
        amountInputChannels = (3 + (2*world.maxAmountOfOffersToOneAgent) )
        numberOfActions = (world.maxAmountOfOffersToOneAgent + 1)
        numberOfNeurons = 16
        super().__init__(world,world.numberOfCores,amountInputChannels,numberOfActions,env.LR_ACTOR, env.LR_CRITIC, env.ACCEPTOR_GAMMA, env.EPS_CLIP,env.ACCEPTOR_K_EPOCHS,numberOfNeurons)

class GloballySharedOfferPPO(GloballySharedPPO):
    def __init__(self,world,env):
        amountInputChannels = ((world.numberOfCores * 2) + 2)
        numberOfActions = ((world.numberOfCores) + 1)
        numberOfNeurons = 16
        super().__init__(world,world.collectionLength,amountInputChannels,numberOfActions,env.LR_ACTOR, env.LR_CRITIC, env.OFFER_GAMMA, env.EPS_CLIP,env.OFFER_K_EPOCHS,numberOfNeurons)

class LocallySharedPPO:
    def __init__(self, world,subLen, amountInputChannels,numberOfActions, lr_actor, lr_critic, gamma, eps_clip, K_epochs,numberOfNeurons):

        self.world = world
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        

        self.amountInputChannels = amountInputChannels
        self.numberOfActions = numberOfActions
        
        self.buffers = [ExperienceBuffer() for _ in range(subLen)]
        self.subLen = subLen
        self.maxSubIndex = subLen - 1

        
        self.policy = ActorCritic(self.amountInputChannels, self.numberOfActions,numberOfNeurons)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(self.amountInputChannels, self.numberOfActions,numberOfNeurons)
        self.policy_old.load_state_dict(self.policy.state_dict())
        

        self.MseLoss = nn.MSELoss()


    def selectAction(self,subID, state):

        with torch.no_grad():
            state = state.float()
            action, action_logprob = self.policy_old.act(state)
        
        #state und action werden hier also schon auf dieser Ebene abgespeichert.
        self.buffers[subID].states.append(state)
        self.buffers[subID].actions.append(action)
        self.buffers[subID].logprobs.append(action_logprob)

        return action.item()
    
    def updateOldPolicy(self):
        self.policy_old.load_state_dict(self.policy.state_dict())

    def update(self,randomIndex,lastUpdate):
        # Monte Carlo estimate of returns
        rewards = deque([])
        discounted_reward = 0
        for reward in reversed(self.buffers[randomIndex].rewards):
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.appendleft(discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffers[randomIndex].states, dim=0)).detach()
        old_actions = torch.squeeze(torch.stack(self.buffers[randomIndex].actions, dim=0)).detach()
        old_logprobs = torch.squeeze(torch.stack(self.buffers[randomIndex].logprobs, dim=0)).detach()

        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        #Bei der letzten Sub-Einheit wird die alte Policy aktualisiert
        if lastUpdate is True:
            for index in range(self.subLen):
                self.buffers[index].clear()
            self.updateOldPolicy()

class LocallySharedAcceptorPPO(LocallySharedPPO):
    def __init__(self,world,env):
        amountInputChannels = (3 + (2*world.maxAmountOfOffersToOneAgent) )
        numberOfActions = (world.maxAmountOfOffersToOneAgent + 1)
        numberOfNeurons = 16
        super().__init__(world,world.numberOfCores,amountInputChannels,numberOfActions,env.LR_ACTOR, env.LR_CRITIC, env.ACCEPTOR_GAMMA, env.EPS_CLIP,env.ACCEPTOR_K_EPOCHS,numberOfNeurons)

class LocallySharedOfferPPO(LocallySharedPPO):
    def __init__(self,world,env):
        amountInputChannels = ((world.numberOfCores * 2) + 2)
        numberOfActions = ((world.numberOfCores) + 1)
        numberOfNeurons = 16
        super().__init__(world,world.collectionLength,amountInputChannels,numberOfActions,env.LR_ACTOR, env.LR_CRITIC, env.OFFER_GAMMA, env.EPS_CLIP,env.OFFER_K_EPOCHS,numberOfNeurons)

