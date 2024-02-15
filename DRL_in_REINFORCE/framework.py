import torch
import torch.nn.functional as F
import numpy as np

from load_config import load_config
from model import Model

class REINFORCE:
    def __init__(self, policy_net, learning_rate, gamma, cuda):
        self.device = torch.device("cuda") if cuda else torch.device("cpu")
        self.policy_net = policy_net.to(self.device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        return

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.policy_net(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample().item()
        return action

    def update(self, transition_dict):
        reward_list = transition_dict['rewards']
        state_list = transition_dict['states']
        action_list = transition_dict['actions']

        G = 0
        self.optimizer.zero_grad()
        for i in reversed(range(len(reward_list))):
            reward = reward_list[i]
            state = torch.tensor([state_list[i]], dtype=torch.float).to(self.device)
            action = torch.tensor([action_list[i]], dtype=torch.int64).view(-1, 1).to(self.device)
            log_prob = torch.log(self.policy_net(state).gather(1, action))
            G = self.gamma * G + reward
            loss = -log_prob * G
            loss.backward()
        self.optimizer.step()
        return loss.cpu().item()
    


# ---TEST
    
if __name__ == "__main__":
    CONFIG        = load_config()
    CUDA          = CONFIG["cuda"]
    AGENT_CONFIG  = CONFIG["agent"]
    LEARNING_RATE = AGENT_CONFIG["lr"]
    GAMMA         = AGENT_CONFIG["gamma"]

    policy_net    = Model(4, 8, 2)
    agent         = REINFORCE(policy_net, LEARNING_RATE, GAMMA, CUDA)

    transition_dict = {
        "states": np.random.randn(64, 4),
        "actions": np.random.randint(0, 2, (64)),
        "rewards": np.random.randn(64),
    }

    loss = agent.update(transition_dict)
    print(loss)