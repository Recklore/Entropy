import math
import random
import torch


class epsilon_greedy:
    def __init__(self, num_c, epsilon_start, epsilon_end, epsilon_decay):
        self.num_actions = num_c
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

    def get_epsilon(self):
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(
            -1.0 * self.steps_done / self.epsilon_decay
        )
        return epsilon

    def select_action(self, state, policy_net, device):
        sample = random.random()
        epsilon_threshold = self.get_epsilon()
        self.steps_done += 1

        if sample > epsilon_threshold:
            with torch.no_grad():
                q_values = policy_net(state.to(device))
                action = q_values.max(1)[1].view(1, 1)
                return action

        else:
            a = random.randrange(self.num_actions)
            return torch.tensor([[a]], device=device, dtype=torch.long)

    def reset(self):
        self.steps_done = 0
