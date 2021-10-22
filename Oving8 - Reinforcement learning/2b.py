import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

env = gym.make('CartPole-v0')


class PolicyNN(nn.Module):
    def __init__(self):
        super(PolicyNN, self).__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        x = self.fc(x)
        return F.softmax(x, dim=1)

def select_action_from_policy(model, state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = model(state)
    m = Categorical(probs)
    action = m.sample()
    return action.item(), m.log_prob(action)

def select_action_from_policy_best(model, state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = model(state)
    if probs[0][0] > probs[0][1]:
        return 0
    else:
        return 1

model = PolicyNN()
optimizer = optim.Adam(model.parameters(), lr=0.01)

def train_simple(num_episodes=10000):
    num_steps = 200
    reward_threshold = 195.0
    ts = []
    for episode in range(num_episodes):
        state = env.reset()
        probs = []
        for t in range(1, num_steps + 1):
            action, prob = select_action_from_policy(model, state)
            probs.append(prob)
            state, _, done, _ = env.step(action)
            if done:
                break
        loss = 0
        for i, prob in enumerate(probs):
            loss += -1 * (t - i) * prob
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ts.append(t)
        avg = sum(ts[-100:])/100.0
        if episode % 100 == 0 and episode > 0:
            print('Episode %i: Average last 100 episodes: %.2f' % (episode, avg))
        if len(ts) > 100 and avg >= reward_threshold:
            print('Stopped at episode %i with an average of %.2f over the last 100' % (episode, avg))
            return

train_simple()
