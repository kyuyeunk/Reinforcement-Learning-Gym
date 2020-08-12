import torch
from torch import nn, optim
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_shape, n_output):
        super().__init__()
        self.n_input = input_shape[0]
        self.n_output = n_output
        self.net = nn.Sequential(
            nn.Linear(self.n_input, 256),
            nn.ReLU(),
            nn.Linear(256, self.n_output)
        )

    def forward(self, x):
        return self.net(x)


class Agent:
    def __init__(self, n_input, n_output, actor_learning_rate, critic_learning_rate, gamma, device):
        self.actor_model = Model(n_input, n_output).to(device)
        self.critic_model = Model(n_input, 1).to(device)

        self.actor_optimizer = optim.Adam(self.actor_model.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic_model.parameters(), lr=critic_learning_rate)

        self.gamma = gamma
        self.log_prob = None

    def decide(self, state):
        prob = F.softmax(self.actor_model(state), dim=-1)
        dist = torch.distributions.Categorical(prob)
        action = dist.sample()

        self.log_prob = torch.log(prob[action])

        return action.item()

    def train(self, state, reward, new_state, done):
        self.actor_model.zero_grad()
        self.critic_model.zero_grad()

        value = self.critic_model(state)
        new_value = self.critic_model(new_state) if not done else 0

        advantage = reward + self.gamma * new_value - value

        actor_loss = -self.log_prob * advantage.item()
        critic_loss = pow(advantage, 2)

        actor_loss.backward()
        critic_loss.backward()

        self.actor_optimizer.step()
        self.critic_optimizer.step()

        return actor_loss, critic_loss
