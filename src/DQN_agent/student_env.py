import torch


# Student environment stimulator
class student_env:
    def __init__(self, dktplus_model, num_c, learning_gain, device):
        self.model = dktplus_model
        self.model.eval()
        self.num_skills = num_c
        self.learning_gain = learning_gain
        self.device = device
        self.reset()

    def reset(self):
        self.q_history = []
        self.r_history = []
        self.state = torch.zeros(1, self.num_skills, device=self.device, dtype=torch.float)

        return self.state

    def step(self, action):
        action = int(action)
        if action < 0 or action >= self.num_skills:
            raise IndexError("action out of range")

        p_before = float(self.state[0, action].item())
        p_after = p_before + (1 - p_before) * self.learning_gain

        response = int(torch.bernoulli(torch.tensor([p_after], device=self.device)).item())

        self.q_history.append(action)
        self.r_history.append(int(response))

        q_tensor = torch.tensor([self.q_history], dtype=torch.long, device=self.device)
        r_tensor = torch.tensor([self.r_history], dtype=torch.long, device=self.device)
        t_tensor = torch.rand(1, len(self.q_history), device=self.device) * 300

        with torch.no_grad():
            dkt_out = torch.sigmoid(self.model(q_tensor, r_tensor, t_tensor))

        next_state = dkt_out[:, -1, :]

        old_mastery = float(self.state.sum().item())
        new_mastery = float(next_state.sum().item())
        mastery_gain = new_mastery - old_mastery

        # ZPD bounus
        if 0.4 <= p_before <= 0.8:
            zpd_bonus = 1.0
        else:
            zpd_bonus = -1.0

        w_mastery, w_zpd = 1.0, 0.5
        norm_mastery_gain = mastery_gain / float(self.num_skills)
        reward = (w_mastery * norm_mastery_gain) + (w_zpd * zpd_bonus)
        reward = torch.tanh(torch.tensor([reward], dtype=torch.float, device=self.device))

        self.state = next_state
        done = bool(self.state.mean().item() > 0.95)

        return self.state, reward, done
