import torch
from torch import nn, optim
import copy


class Model(nn.Module):
    def __init__(self, layers):
        super().__init__()
        assert(len(layers) >= 2)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def merge_samples(samples):
    states = torch.cat([s.state for s in samples])
    rewards = torch.cat([s.reward for s in samples])
    actions = torch.cat([s.action for s in samples])
    next_states = torch.cat([s.next_state for s in samples])

    mask = torch.cat([s.done for s in samples])
    mask = (mask == False)

    return states, rewards, actions, next_states, mask


class Agent:
    def __init__(self, layers, learning_rate, gamma, device):
        self.train_model = Model(layers).to(device)
        self.target_model = copy.deepcopy(self.train_model)
        self.target_model.eval()
        self.update_target_model()

        self.optimizer = optim.Adam(self.train_model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.gamma = gamma

    def update_target_model(self):
        self.target_model.load_state_dict(self.train_model.state_dict())

    def predict(self, state):
        q_val = self.train_model(state)[0]
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

    def save_model(self, game, timestamp):
        time_str = timestamp.strftime("%y%m%d-%H%M")
        torch.save(self.target_model.state_dict(), 'runs/{}_dqn_{}/target.pt'.format(game, time_str))
        torch.save(self.train_model.state_dict(), 'runs/{}_dqn_{}/train.pt'.format(game, time_str))

    def load_model(self, game, timestamp):
        self.target_model.load_state_dict(torch.load('runs/{}_dqn_{}/target.pt'.format(game, timestamp)))
        self.train_model.load_state_dict(torch.load('runs/{}_dqn_{}/train.pt'.format(game, timestamp)))
