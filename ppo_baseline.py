import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import gym
import numpy as np
from numpy.core.fromnumeric import clip

from torch.utils import tensorboard
w = tensorboard.SummaryWriter()
device = torch.device('cpu')



class RolloutBuffer:
    def __init__(self):
        self.actions = [] 
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, is_continuous, action_std_init=0):
        super(ActorCritic, self).__init__()

        self.is_continuous = is_continuous
        if is_continuous:
            self.action_dim = action_dim
            self.action_std = action_std_init
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

        
        if is_continuous:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64,64),
                nn.Tanh(),
                nn.Linear(64,action_dim),
                nn.Tanh()
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.Tanh(),
                nn.Linear(128,128),
                nn.Tanh(),
                nn.Linear(128,action_dim),
                nn.Softmax(dim=-1)
            )

        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64,1)
        )
    def set_action_std(self, new_std):
        self.action_var = torch.full((self.action_dim,), new_std*new_std).to(device)

    def foward(self):
        raise NotImplementedError

    def act(self, state):
        if self.is_continuous:
            action = self.actor(state)
            cov = torch.diag(self.action_var).unsqueeze(0)
            dist = MultivariateNormal(action, cov)

        else:
            action = self.actor(state)
            dist = Categorical(action)

        action = dist.sample()
        action_log = dist.log_prob(action)

        return action, action_log
    def crt(self, state):
        value = self.critic(state)
        if self.is_continuous:

            action = self.actor(state)
            cov = torch.diag(self.action_var).unsqueeze(0)
            dist = MultivariateNormal(action, cov)
            dist_entropy = dist.entropy().mean()

        return value, dist_entropy
    def decay_std(self, rate_, min_):
        self.action_std = self.action_std - rate_
        self.action_std = round(self.action_std, 4)
        if (self.action_std <= min_):
                self.action_std = min_
        self.action_var = torch.full((self.action_dim,), self.action_std * self.action_std).to(device)

        
def loss(old_log_prob, new_log_prob, advantage, eps):
    ratio = (new_log_prob - old_log_prob).exp()
    surr1 = ratio * advantage
    surr2 = torch.clamp(ratio, 1-eps, 1+eps)* advantage
    m = torch.min(surr1, surr2)
    return -m



# Lets start the training
random_seed = 1234



# env = gym.make('CartPole-v1')

env = gym.make('LunarLanderContinuous-v2')
is_con = 1
# cont. config
action_std = 0.6
action_std_decay_rate = 0.05
min_action_std = 0.1
action_std_decay_freq = int(50)
learning_rate = 0.001


state_dim = env.observation_space.shape[0]

if is_con:
    n_actions = env.action_space.shape[0]
else:
    n_actions = env.action_space.n

ac = ActorCritic(state_dim, n_actions, is_con, action_std)
adam_actor = torch.optim.Adam(ac.actor.parameters(), lr=0.0003)
adam_critic= torch.optim.Adam(ac.critic.parameters(), lr=0.0003)


def loss(old_log_prob, new_log_prob, advantage, eps):
    ratio = (new_log_prob - old_log_prob).exp()
    clipped = torch.clamp(ratio, 1-eps, 1+eps)* advantage
    m = torch.min(ratio*advantage, clipped).mean()
    return -m



episode_rewards = []
gamma = 0.98
eps = 0.3
s = 0
mode = "ppo"
ent_coef = 0.001
grad_clip = 0.5

def t(x): return torch.from_numpy(x).float()

for i in range(1000):
    prev_prob_act = None
    done = False
    total_reward = 0 
    state = env.reset()

    while not done:
        s+=1
        action, action_prob = ac.act(t(state))


        if is_con:
            next_state, reward, done, _ = env.step(action.detach().data.numpy().flatten())

        else:
            next_state, reward, done, _ = env.step(action.detach().data.numpy())


        old_value, entropy  = ac.crt(t(state))
        new_value, _ =ac.crt(t(next_state))

        advantage = reward + (1-done)*gamma*new_value- old_value
        w.add_scalar("loss/advantage", advantage, global_step = s)
        total_reward += reward
        state = next_state

        if prev_prob_act:
            actor_loss = loss(prev_prob_act.detach(), action_prob, advantage.detach(), eps) - ent_coef*entropy
            w.add_scalar("loss/actor_loss", actor_loss, global_step=s)
            adam_actor.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_value_(ac.actor.parameters(), grad_clip)

            w.add_histogram("gradients/actor",
                             torch.cat([p.grad.view(-1) for p in ac.actor.parameters()]), global_step=s)
            adam_actor.step()

            critic_loss = 0.5* advantage.pow(2).mean()#- ent_coef*entropy
            w.add_scalar("loss/critic_loss", critic_loss, global_step=s)
            adam_critic.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_value_(ac.critic.parameters(), grad_clip)
            w.add_histogram("gradients/critic",
                             torch.cat([p.grad.view(-1) for p in ac.critic.parameters()]), global_step=s)

            adam_critic.step()

        prev_prob_act = action_prob

        if is_con and i%action_std_decay_freq==0:
            ac.decay_std(action_std_decay_rate, min_action_std)
    print(total_reward)
    w.add_scalar("reward/episode_reward", total_reward, global_step=i)
    episode_rewards.append(total_reward)
