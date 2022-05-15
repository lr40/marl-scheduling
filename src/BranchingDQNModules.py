import random, numpy as np, torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F

class AgentConfig:
    def __init__(self, 
                 epsilon_start = 1.,
                 epsilon_final = 0.01,
                 epsilon_decay = 8000,
                 gamma = 0.99, 
                 lr = 1e-4, 
                 target_net_update_freq = 1000, 
                 memory_size = 100000, 
                 batch_size = 128, 
                 learning_starts = 5000,
                 max_frames = 10000000): 

        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.epsilon_by_frame = lambda i: self.epsilon_final + (self.epsilon_start - self.epsilon_final) * np.exp(-1. * i / self.epsilon_decay)

        self.gamma =gamma
        self.lr =lr

        self.target_net_update_freq =target_net_update_freq
        self.memory_size =memory_size
        self.batch_size =batch_size

        self.learning_starts = learning_starts
        self.max_frames = max_frames


class ExperienceReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        
        batch = random.sample(self.memory, batch_size)
        states = []
        actions = []
        rewards = []
        next_states = [] 
        dones = []

        for b in batch: 
            states.append(b[0])
            actions.append(b[1])
            rewards.append(b[2])
            next_states.append(b[3])
            dones.append(b[4])


        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)

class BranchingQNetwork(nn.Module):

    def __init__(self, obs, ac_dim, n): 

        super().__init__()

        self.ac_dim = ac_dim
        self.n = n 

        self.model = nn.Sequential(nn.Linear(obs, 128), 
                                   nn.ReLU(),
                                   nn.Linear(128,128), 
                                   nn.ReLU())

        self.value_head = nn.Linear(128, 1)
        self.adv_heads = nn.ModuleList([nn.Linear(128, n) for i in range(ac_dim)])

    def forward(self, x): 

        out = self.model(x)
        value = self.value_head(out)
        advs = torch.stack([l(out) for l in self.adv_heads], dim = 1)

        # print(advs.shape)
        # print(advs.mean(2).shape)
        test =  advs.mean(2, keepdim = True)
        # input(test.shape)
        q_val = value.unsqueeze(2) + advs - advs.mean(2, keepdim = True )
        # input(q_val.shape)

        return q_val

class BranchingDQN(nn.Module): 

    def __init__(self, obs, ac, n, config): 

        super().__init__()

        self.q = BranchingQNetwork(obs, ac,n )
        self.target = BranchingQNetwork(obs, ac,n )

        self.target.load_state_dict(self.q.state_dict())

        self.target_net_update_freq = config.target_net_update_freq
        self.update_counter = 0

    def get_action(self, x): 

        with torch.no_grad(): 
            # a = self.q(x).max(1)[1]
            out = self.q(x).squeeze(0)
            action = torch.argmax(out, dim = 1)
        return action.numpy()

    def update_policy(self, adam, memory, params): 

        b_states, b_actions, b_rewards, b_next_states, b_masks = memory.sample(params.batch_size)

        states = torch.tensor(b_states).float()
        actions = torch.tensor(b_actions).long().reshape(states.shape[0],-1,1)
        rewards = torch.tensor(b_rewards).float().reshape(-1,1)
        next_states = torch.tensor(b_next_states).float()
        masks = torch.tensor(b_masks).float().reshape(-1,1)

        qvals = self.q(states)


        current_q_values = self.q(states).gather(2, actions).squeeze(-1)

        with torch.no_grad():


            argmax = torch.argmax(self.q(next_states), dim = 2)

            max_next_q_vals = self.target(next_states).gather(2, argmax.unsqueeze(2)).squeeze(-1)
            max_next_q_vals = max_next_q_vals.mean(1, keepdim = True)

        expected_q_vals = rewards + max_next_q_vals*0.99*masks
        # print(expected_q_vals[:5])
        loss = F.mse_loss(expected_q_vals, current_q_values)

        # input(loss)

        # print('\n'*5)
        
        adam.zero_grad()
        loss.backward()

        for p in self.q.parameters(): 
            p.grad.data.clamp_(-1.,1.)
        adam.step()

        self.update_counter += 1
        if self.update_counter % self.target_net_update_freq == 0: 
            self.update_counter = 0 
            self.target.load_state_dict(self.q.state_dict())




env = ...
        
config = AgentConfig()
memory = ExperienceReplayMemory(config.memory_size)
agent = BranchingDQN(env.observation_space.shape[0], env.action_space.shape[0], config)
adam = optim.Adam(agent.q.parameters(), lr = config.lr) 


s = env.reset()
ep_reward = 0. 
recap = []

for frame in range(config.max_frames): 

    epsilon = config.epsilon_by_frame(frame)

    if np.random.random() > epsilon: 
        action = agent.get_action(s)
    else: 
        action = np.random.randint(0, size = env.action_space.shape[0])

    ns, r, done, infos = env.step(action)
    ep_reward += r 

    if done:
        ns = env.reset()
        recap.append(ep_reward)
        ep_reward = 0.  

    memory.push((s.reshape(-1).numpy().tolist(), action, r, ns.reshape(-1).numpy().tolist(), 0. if done else 1.))
    s = ns  


    if frame > config.learning_starts:
        agent.update_policy(adam, memory, config)

    


