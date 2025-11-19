import random
from Action import ActionType

class LearningPolicy:
    """
    Implementação simples estilo Q-learning.
    """
    def __init__(self, actions=None, alpha=0.2, gamma=0.95, epsilon=0.1):
        if actions is None:
            actions = list(ActionType)

        self.actions = actions
        self.q = {}  # {(state): {action: value}}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_state_key(self, observation):
        return tuple(sorted(observation.info.items()))

    def choose_action(self, observation):
        state = self.get_state_key(observation)
        if random.random() < self.epsilon or state not in self.q:
            return random.choice(self.actions)
        return max(self.q[state], key=self.q[state].get)

    def update(self, obs, action, reward, next_obs):
        state = self.get_state_key(obs)
        next_state = self.get_state_key(next_obs)

        if state not in self.q:
            self.q[state] = {a: 0.0 for a in self.actions}
        if next_state not in self.q:
            self.q[next_state] = {a: 0.0 for a in self.actions}

        q_old = self.q[state][action]
        q_max_next = max(self.q[next_state].values())

        self.q[state][action] = q_old + self.alpha * (reward + self.gamma * q_max_next - q_old)
