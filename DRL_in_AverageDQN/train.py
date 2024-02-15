import os
import gym
import matplotlib.pyplot as plt
from tqdm import tqdm

from load_config import load_config
from ReplayBuffer import ReplayBuffer
from model import AverageModel
from framework import AverageDQN
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
EPSILON       = AGENT_CONFIG["epsilon"]
TARGET_UPDATE = AGENT_CONFIG["target_update"]
EPOCHS        = TRAIN_CONFIG["epochs"]
REWARD_IMG    = os.path.join(logger.root, SHOW_CONFIG["reward_img"])

def test(env_name : str, agent : AverageDQN):
    env = gym.make(env_name, new_step_api=True, render_mode="human")
    test_reward = 0
    state = env.reset()
    done = False
    while not done:
        env.render()
        action = agent.take_action(state, train=False)
        next_state, reward, _, done, _ = env.step(action)
        state = next_state
        test_reward += reward
    logger.info(f"Test reward:{test_reward}")
    env.close()
    return

def train(env_name : str, replay_buffer : ReplayBuffer, agent : AverageDQN, epochs : int):
    env = gym.make(env_name, new_step_api=True)
    epoch_list   = []
    reward_list = []
    
    for epoch in tqdm(range(1, epochs + 1)):
        episode_reward = 0
        state = env.reset()
        done = False
        while not done:
            action = agent.take_action(state)
            next_state, reward, _, done, _ = env.step(action)
            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            if replay_buffer.can_sample():
                b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample()
                transition_dict = {
                    'states': b_s,
                    'actions': b_a,
                    'next_states': b_ns,
                    'rewards': b_r,
                    'dones': b_d
                }
                agent.update(transition_dict)
        
        if epoch % 50 == 0:
            logger.save_model(agent.q_net, f"_{epoch}.pth")
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

    q_net         = AverageModel(state_dim, 64, action_dim, 4)
    target_q_net  = AverageModel(state_dim, 64, action_dim, 4)
    buffer        = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, MINIMAL_SIZE)
    agent         = AverageDQN(q_net, target_q_net, 2, LEARNING_RATE, GAMMA, EPSILON, TARGET_UPDATE, CUDA)

    epoch_list, reward_list = train(ENV_NAME, buffer, agent, EPOCHS)
    draw(epoch_list, reward_list)

    test(ENV_NAME, agent)