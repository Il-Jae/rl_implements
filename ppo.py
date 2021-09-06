import gym
import numpy as np
from numpy.core.fromnumeric import clip

import torch
import torch.nn as nn

from torch.utils import tensorboard
w = tensorboard.SummaryWriter()

# class ToyStockGym(gym.Env):
#     def __init__(self):
#         self.reset()

#     def step(self, actions):
#         reward, ended = self.env.step(actions)


env = gym.make('CartPole-v1')


class ActorCritic(nn.Module):
    def __init__(self, state_dim, n_actions, activation=nn.Tanh):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            activation(),
            nn.Linear(64,32),
            activation(),
            nn.Linear(32,n_actions),
            nn.Softmax()
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            activation(),
            nn.Linear(64,32),
            activation(),
            nn.Linear(32,1)
        )
    def forward(self, X):
        return self.actor(X), self.critic(X)

state_dim = env.observation_space.shape[0]
n_actions = env.action_space.n

ac = ActorCritic(state_dim, n_actions)
optimizer = torch.optim.Adam(ac.parameters(), lr=3e-4)


def loss(old_log_prob, new_log_prob, advantage, eps, mode):
    ratio = (new_log_prob - old_log_prob).exp()
    if (mode=='ppo'):
        clipped = torch.clamp(ratio, 1-eps, 1+eps)* advantage
    # else:
    #     clipped = ratio*advantage

    m = torch.min(ratio*advantage, clipped)
    return -m



episode_rewards = []
gamma = 0.98
eps = 0.1
s = 0
max_grad_norm =0.5
mode = "ppo"


def t(x): return torch.from_numpy(x).float()

for i in range(800):
    prev_prob_act = None
    done = False
    total_reward = 0 
    state = env.reset()

    while not done:
        s+=1
        probs = actor(t(state))
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()
        prob_act = dist.log_prob(action)

        next_state, reward, done, info = env.step(action.detach().data.numpy())
        advantage = reward + (1-done)*gamma*critic(t(next_state)) - critic(t(state))

        w.add_scalar("loss/advantage", advantage, global_step = s)
        w.add_scalar("actions/action_0_prob", dist.probs[0], global_step=s)
        w.add_scalar("actions/action_1_prob", dist.probs[1], global_step=s)

        total_reward += reward
        state = next_state

        if prev_prob_act:
            actor_loss = loss(prev_prob_act.detach(), prob_act, advantage.detach(), eps, mode)
            w.add_scalar("loss/actor_loss", actor_loss, global_step=s)
            adam_actor.zero_grad()
            actor_loss.backward()
            w.add_histogram("gradients/actor",
                             torch.cat([p.grad.view(-1) for p in actor.parameters()]), global_step=s)
            adam_actor.step()


            critic_loss = advantage.pow(2).mean()
            w.add_scalar("loss/critic_loss", critic_loss, global_step=s)
            adam_critic.zero_grad()
            critic_loss.backward()
            w.add_histogram("gradients/critic",
                             torch.cat([p.data.view(-1) for p in critic.parameters()]), global_step=s)
            adam_critic.step()

        prev_prob_act = prob_act
    print(total_reward)
    w.add_scalar("reward/episode_reward", total_reward, global_step=i)
    episode_rewards.append(total_reward)