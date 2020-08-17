import torch
from torch import nn, optim


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


def merge_samples(samples):
    states = torch.stack([s.state for s in samples])
    rewards = torch.stack([s.reward for s in samples])
    actions = torch.stack([s.action for s in samples])
    next_states = torch.stack([s.next_state for s in samples])

    mask = torch.stack([s.done for s in samples])
    mask = (mask == False)

    return states, rewards, actions, next_states, mask


class Agent:
    def __init__(self, layers, learning_rate, gamma, device):
        self.train_model = Model(layers).to(device)
        self.target_model = Model(layers).to(device)
        self.target_model.eval()
        self.update_target_model()

        self.optimizer = optim.Adam(self.train_model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.gamma = gamma

    def update_target_model(self):
        self.target_model.load_state_dict(self.train_model.state_dict())

    def predict(self, state):
        q_val = self.train_model(state)
        return torch.argmax(q_val).item()

    def train(self, samples):
        states, rewards, actions, next_states, mask = merge_samples(samples)

        self.optimizer.zero_grad()
        train_q_val = self.train_model(states).gather(dim=1, index=actions)

        target_q_val_next = self.target_model(next_states).max(dim=1)[0].unsqueeze(-1) * mask
        target_q_val = rewards + self.gamma * target_q_val_next

        loss = self.criterion(train_q_val, target_q_val)
        loss.backward()
        self.optimizer.step()

        return loss
