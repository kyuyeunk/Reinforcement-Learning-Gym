import torch
from torch import nn, optim
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, layers):
        super().__init__()
        assert(len(layers) >= 2)

        networks = []
        for i in range(0, len(layers) - 2):
            networks.append(nn.Linear(layers[i], layers[i+1]))
            networks.append(nn.ReLU())
        networks.append(nn.Linear(layers[-2], layers[-1]))

        self.net = nn.Sequential(*networks)

    def forward(self, x):
        return self.net(x)


class Agent:
    def __init__(self, actor_layers, critic_layers, actor_learning_rate, critic_learning_rate, gamma, device):
        self.actor_model = Model(actor_layers).to(device)
        self.critic_model = Model(critic_layers).to(device)

        self.actor_optimizer = optim.Adam(self.actor_model.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic_model.parameters(), lr=critic_learning_rate)

        self.gamma = gamma

    def prob(self, state):
        return F.softmax(self.actor_model(state), dim=-1)

    def train(self, state, action_prob, reward, new_state, done):
        self.actor_model.zero_grad()
        self.critic_model.zero_grad()

        value = self.critic_model(state)
        new_value = self.critic_model(new_state) if not done else 0

        advantage = reward + self.gamma * new_value - value

        actor_loss = -torch.log(action_prob) * advantage.item()
        critic_loss = pow(advantage, 2)

        actor_loss.backward()
        critic_loss.backward()

        self.actor_optimizer.step()
        self.critic_optimizer.step()

        return actor_loss, critic_loss
