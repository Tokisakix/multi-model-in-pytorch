import torch
import torch.nn.functional as F
import numpy as np

from load_config import load_config
from model import Model

class DQN:
    def __init__(self, q_net, target_q_net, action_dim, learning_rate, gamma, epsilon, target_update, cuda):
        self.action_dim = action_dim
        self.device = torch.device("cuda") if cuda else torch.device("cpu")
        self.q_net = q_net.to(self.device)
        self.target_q_net = target_q_net.to(self.device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0
        return

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.int64).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())
        self.count += 1
        return dqn_loss.cpu().item()
    


# ---TEST
    
if __name__ == "__main__":
    CONFIG        = load_config()
    CUDA          = CONFIG["cuda"]
    AGENT_CONFIG  = CONFIG["agent"]
    LEARNING_RATE = AGENT_CONFIG["lr"]
    GAMMA         = AGENT_CONFIG["gamma"]
    EPSILON       = AGENT_CONFIG["epsilon"]
    TARGET_UPDATE = AGENT_CONFIG["target_update"]

    q_net         = Model(4, 8, 2)
    target_q_net  = Model(4, 8, 2)
    agent         = DQN(q_net, target_q_net, 2, LEARNING_RATE, GAMMA, EPSILON, TARGET_UPDATE, CUDA)

    transition_dict = {
        "states": np.random.randn(64, 4),
        "actions": np.random.randint(0, 2, (64)),
        "next_states": np.random.randn(64, 4),
        "rewards": np.random.randn(64),
        "dones": [False] * 64,
    }

    loss = agent.update(transition_dict)
    print(loss)