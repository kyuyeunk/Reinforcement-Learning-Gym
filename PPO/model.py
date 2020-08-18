import torch
from torch import nn, optim
import torch.nn.functional as F
from shared.utils import data_to_torch


class History:
    def __init__(self, state, action, action_prob, reward, next_state, done):
        self.state = state
        self.action = action
        self.action_prob = action_prob
        self.reward = reward
        self.next_state = next_state
        self.done = done


class Model(nn.Module):
    def __init__(self, layers):
        super().__init__()
        assert(len(layers) >= 2)

        networks = []
        for i in range(0, len(layers) - 2):
            networks.append(nn.Linear(layers[i], layers[i+1]))
            networks.append(nn.ReLU())

        actor_networks = networks + [nn.Linear(layers[-2], layers[-1])]
        self.actor_net = nn.Sequential(*actor_networks)

        critic_networks = networks + [nn.Linear(layers[-2], 1)]
        self.critic_net = nn.Sequential(*critic_networks)

    def forward(self, x):
        return self.actor_net(x)

    def value(self, x):
        return self.critic_net(x)


class Agent:
    def __init__(self, layers, learning_rate, gamma, lmbda, eps, k, batch_size, device):
        self.model = Model(layers).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.gamma = gamma
        self.lmbda = lmbda
        self.eps = eps
        self.k = k

        self.batch_size = batch_size
        self.buffer = []

        self.device = device

    def prob(self, state):
        return F.softmax(self.model(state), dim=-1)

    def train(self):
        actor_loss = 0
        critic_loss = 0
        states, last_state, actions, actions_prob, rewards, mask = self.merge_samples()

        for _ in range(self.k):
            all_states = torch.cat((states, last_state.unsqueeze(0)))
            all_values = self.model.value(all_states)

            values = all_values[:-1]
            next_values = all_values[1:]

            target_values = rewards + self.gamma * next_values * mask
            deltas = target_values - values

            advantage = 0
            advantages = []
            for delta in reversed(deltas):
                advantage = delta.item() + self.gamma * self.lmbda * advantage
                advantages.append([advantage])

            advantages.reverse()
            advantages = data_to_torch(advantages, torch.float32, self.device)

            new_actions_prob = self.prob(states).gather(1, actions)

            ratio = torch.exp(torch.log(new_actions_prob) - torch.log(actions_prob))
            lhs = ratio * advantages
            rhs = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantages

            actor_loss = -torch.min(lhs, rhs).mean()
            critic_loss = F.mse_loss(values, target_values)

            loss = actor_loss + critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return actor_loss, critic_loss

    def store_history(self, state, action, action_prob, reward, next_state, done):
        tmp = History(state, action, action_prob, reward, next_state, done)
        self.buffer.append(tmp)

    def reset_history(self):
        self.buffer = []

    def merge_samples(self):
        samples = self.buffer
        states = torch.stack([s.state for s in samples])
        actions = torch.stack([s.action for s in samples])
        actions_prob = torch.stack([s.action_prob for s in samples])
        rewards = torch.stack([s.reward for s in samples])
        last_state = samples[-1].next_state

        mask = torch.stack([s.done for s in samples])
        mask = (mask == False)

        self.reset_history()
        return states, last_state, actions, actions_prob, rewards, mask

    def is_buffer_full(self):
        return len(self.buffer) == self.batch_size

    def is_buffer_empty(self):
        return len(self.buffer) == 0

