import torch
import random
import numpy as np
from itertools import count
from collections import namedtuple

from src.DQN_agent.dqn import DQN
from src.DQN_agent.replay_memory import replay_memory, Transition
from src.DQN_agent.policy import epsilon_greedy
from src.DQN_agent.student_env import student_env

# Model Paths
DQN_MODEL_PATH = "./models/DQN_agent.pt"
DKT_MODEL_PATH = "./models/DKT_model.pt"

# Model Hyperparameters
NUM_EPISODES = 600
TARGET_UPDATE_EPISODES = 20
MAX_STEPS = 200
BATCH_SIZE = 128
SEED = 7

LEARNING_GAIN = 0.2
LEARNING_RATE = 0.0001
HIDDEN_SIZE = 256
MEMORY_CAPACITY = 10000
GAMMA = 0.9  # Discount factor
NUM_SKILLS = 44
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000


# Seed For Reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Loading Traced DKTplus Model
try:
    dkt_model = torch.jit.load(DKT_MODEL_PATH, map_location=DEVICE)
    print("Successfully loaded the DKT+ mode")
except Exception as e:
    print("Error in loading the DKT+ model: ", e)


# Initailisation
policy_net = DQN(NUM_SKILLS, HIDDEN_SIZE).to(DEVICE)
target_net = DQN(NUM_SKILLS, HIDDEN_SIZE).to(DEVICE)
target_net.load_state_dict(policy_net.state_dict())

env = student_env(dkt_model, NUM_SKILLS, LEARNING_GAIN, DEVICE)
epsilon_policy = epsilon_greedy(NUM_SKILLS, EPS_START, EPS_END, EPS_DECAY)

optimizer = torch.optim.Adam(policy_net.parameters(), lr=LEARNING_RATE, amsgrad=True)
memory = replay_memory(MEMORY_CAPACITY)


# Training Loop
episode_duration = []

for episode_i in range(NUM_EPISODES):
    state = env.reset()

    for t in count():
        action = epsilon_policy.select_action(state, policy_net, DEVICE)
        next_state, reward, done = env.step(action.item())

        reward = reward.to(DEVICE)
        if not done:
            next_state = next_state.to(DEVICE)
        else:
            next_state = None

        memory.push(state, action, next_state, reward, done)
        state = next_state

        if len(memory) >= BATCH_SIZE:
            transition = memory.sample(BATCH_SIZE)
            batch = Transition(*zip(*transition))

            non_final_mask = torch.tensor([d is False for d in batch.done], device=DEVICE, dtype=torch.bool)
            non_final_next_states = (
                torch.cat([s for s, d in zip(batch.next_state, batch.done) if not d])
                if any(not d for d in batch.done)
                else torch.empty((0, NUM_SKILLS), device=DEVICE)
            )

            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)

            state_action_values = policy_net(state_batch).gather(1, action_batch)

            next_state_values = torch.zeros(BATCH_SIZE, device=DEVICE)

            with torch.no_grad():
                next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]

            expected_state_action_values = (next_state_values * GAMMA) + reward_batch

            criterion = torch.nn.SmoothL1Loss()
            loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
            optimizer.step()

        if (episode_i + 1) % TARGET_UPDATE_EPISODES == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if done or t > MAX_STEPS:
            episode_duration.append(t + 1)
            print(f"Episode: {episode_i} finished after {t+1} steps")
            break

print("Training Completed")

policy_net.eval()

dummy_input = torch.zeros(1, NUM_SKILLS, device=DEVICE)

dqn = torch.jit.trace(policy_net, dummy_input)
torch.jit.save(dqn, DQN_MODEL_PATH)

print("DQN policy model saved successfully")
