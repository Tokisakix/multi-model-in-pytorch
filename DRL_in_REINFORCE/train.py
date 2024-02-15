import os
import gym
import matplotlib.pyplot as plt
from tqdm import tqdm

from load_config import load_config
from model import Model
from framework import REINFORCE
from logger import Logger

CONFIG        = load_config()
CUDA          = CONFIG["cuda"]
ENV_NAME      = CONFIG["env"]
LOG_CONFIG    = CONFIG["log"]
BUFFER_CONFIG = CONFIG["buffer"]
AGENT_CONFIG  = CONFIG["agent"]
TRAIN_CONFIG  = CONFIG["train"]
SHOW_CONFIG   = CONFIG["show"]
LOG_ROOT      = LOG_CONFIG["root"]
SAVE_NUM      = LOG_CONFIG["save_num"]
logger        = Logger(LOG_ROOT, SAVE_NUM)

BUFFER_SIZE   = BUFFER_CONFIG["buffer_size"]
BATCH_SIZE    = BUFFER_CONFIG["batch_size"]
MINIMAL_SIZE  = BUFFER_CONFIG["minimal_size"]
LEARNING_RATE = AGENT_CONFIG["lr"]
GAMMA         = AGENT_CONFIG["gamma"]
EPOCHS        = TRAIN_CONFIG["epochs"]
REWARD_IMG    = os.path.join(logger.root, SHOW_CONFIG["reward_img"])

def test(env_name : str, agent : REINFORCE):
    env = gym.make(env_name, new_step_api=True, render_mode="human")
    test_reward = 0
    state = env.reset()
    done = False
    while not done:
        env.render()
        action = agent.take_action(state)
        next_state, reward, _, done, _ = env.step(action)
        state = next_state
        test_reward += reward
    logger.info(f"Test reward:{test_reward}")
    env.close()
    return

def train(env_name : str, agent : REINFORCE, epochs : int):
    env = gym.make(env_name, new_step_api=True)
    epoch_list   = []
    reward_list = []
    
    for epoch in tqdm(range(1, epochs + 1)):
        episode_reward = 0
        transition_dict = {
            'states': [],
            'actions': [],
            'next_states': [],
            'rewards': [],
            'dones': []
        }
        state = env.reset()
        done = False
        while not done:
            action = agent.take_action(state)
            next_state, reward, done, _, _ = env.step(action)
            transition_dict['states'].append(state)
            transition_dict['actions'].append(action)
            transition_dict['next_states'].append(next_state)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(done)
            state = next_state
            episode_reward += reward
        agent.update(transition_dict)
        
        if epoch % 50 == 0:
            logger.save_model(agent.policy_net, f"_{epoch}.pth")
            logger.info(f"Epoch:{epoch} Rewards:{episode_reward}")
            logger.info(f"Save model in _{epoch}.pth")
        epoch_list.append(epoch)
        reward_list.append(episode_reward)

    logger.info("Finished training!")
    env.close()
    return epoch_list, reward_list

def draw(epoch_list, reward_list):
    plt.plot(epoch_list, reward_list, label="Reward")
    plt.title("Reward Img")
    plt.legend()
    plt.savefig(REWARD_IMG)
    plt.show()
    plt.close()
    return



# ---TEST

if __name__ == "__main__":
    env = gym.make(ENV_NAME, new_step_api=True)
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.n
    env.close()

    policy_net    = Model(state_dim, 64, action_dim)
    agent         = REINFORCE(policy_net, LEARNING_RATE, GAMMA, CUDA)

    epoch_list, reward_list = train(ENV_NAME, agent, EPOCHS)
    draw(epoch_list, reward_list)

    test(ENV_NAME, agent)