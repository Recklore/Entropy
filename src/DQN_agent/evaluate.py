import torch
import numpy as np
import random

from src.DQN_agent.dqn import DQN
from src.DQN_agent.student_env import student_env

# --- Configuration ---
# Model Paths
DQN_MODEL_PATH = "./models/DQN_agent.pt"
DKT_MODEL_PATH = "./models/DKT_model.pt"

# Evaluation Hyperparameters
NUM_EVAL_EPISODES = 50
MAX_STEPS_PER_EPISODE = 200
SEED = 7

# Student Environment & Model Parameters
LEARNING_GAIN = 0.2
NUM_SKILLS = 44
DQN_HIDDEN_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Policies ---


class LowestMasteryPolicy:
    """A baseline policy that selects the skill with the lowest mastery."""

    def select_action(self, state):
        # state is the mastery vector, shape (1, num_skills)
        # Select the skill index with the lowest mastery
        action = torch.argmin(state, dim=1)
        return action.view(1, 1)


class DqnPolicy:
    """The trained DQN policy."""

    def __init__(self, model_path):
        self.model = torch.jit.load(model_path, map_location=DEVICE)
        self.model.eval()

    def select_action(self, state):
        with torch.no_grad():
            # self.model(state) will return Q-values for each action
            # We select the action with the highest Q-value
            action = self.model(state).max(1)[1]
            return action.view(1, 1)


# --- Evaluation Function ---


def evaluate_policy(policy, env, num_episodes, max_steps):
    """Evaluates a given policy in the student environment."""
    total_rewards = []
    total_mastery_gains = []

    for i in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        initial_mastery = state.sum().item()

        for t in range(max_steps):
            action = policy.select_action(state)
            next_state, reward, done = env.step(action.item())

            state = next_state
            episode_reward += reward.item()

            if done:
                break

        final_mastery = state.sum().item()
        mastery_gain = final_mastery - initial_mastery

        total_rewards.append(episode_reward)
        total_mastery_gains.append(mastery_gain)

    avg_reward = np.mean(total_rewards)
    avg_mastery_gain = np.mean(total_mastery_gains)

    return avg_reward, avg_mastery_gain


if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    try:
        dkt_model = torch.jit.load(DKT_MODEL_PATH, map_location=DEVICE)
        print("Successfully loaded the DKT+ model for the environment.")
    except Exception as e:
        print(f"Fatal: Error loading the DKT+ model from {DKT_MODEL_PATH}. Exception: {e}")
        exit()

    env = student_env(dkt_model, NUM_SKILLS, LEARNING_GAIN, DEVICE)

    print("\n--- Evaluating DQN Agent ---")
    try:
        dqn_policy = DqnPolicy(DQN_MODEL_PATH)
        dqn_avg_reward, dqn_avg_mastery_gain = evaluate_policy(
            dqn_policy, env, NUM_EVAL_EPISODES, MAX_STEPS_PER_EPISODE
        )
        print("DQN Agent evaluation complete.")
    except Exception as e:
        print(f"Fatal: Error evaluating DQN agent. Is the model at {DQN_MODEL_PATH}? Exception: {e}")
        exit()

    # Evaluate Baseline Agent (Lowest Mastery)
    print("\n--- Evaluating Baseline (Lowest Mastery) Agent ---")
    try:
        baseline_policy = LowestMasteryPolicy()
        baseline_avg_reward, baseline_avg_mastery_gain = evaluate_policy(
            baseline_policy, env, NUM_EVAL_EPISODES, MAX_STEPS_PER_EPISODE
        )
        print("Baseline Agent evaluation complete.")
    except Exception as e:
        print(f"Fatal: Error evaluating Baseline agent. Exception: {e}")
        exit()

    print("\n\n--- Evaluation Summary ---")
    print(f"Number of Episodes per Agent: {NUM_EVAL_EPISODES}")
    print(f"Max Steps per Episode: {MAX_STEPS_PER_EPISODE}\n")

    print(f"DQN Agent:")
    print(f"  - Average Reward: {dqn_avg_reward:.4f}")
    print(f"  - Average Mastery Gain: {dqn_avg_mastery_gain:.4f}\n")

    print(f"Baseline (Lowest Mastery) Agent:")
    print(f"  - Average Reward: {baseline_avg_reward:.4f}")
    print(f"  - Average Mastery Gain: {baseline_avg_mastery_gain:.4f}\n")

    reward_diff = dqn_avg_reward - baseline_avg_reward
    mastery_diff = dqn_avg_mastery_gain - baseline_avg_mastery_gain

    print("--- Comparison ---")
    print(f"Reward Improvement (DQN vs Baseline): {reward_diff:+.4f} ({reward_diff/abs(baseline_avg_reward):+.2%})")
    print(
        f"Mastery Gain Improvement (DQN vs Baseline): {mastery_diff:+.4f} ({mastery_diff/abs(baseline_avg_mastery_gain):+.2%})"
    )
    print("\nEvaluation finished.")
