from Agent import Agent
from Action import Action, ActionType
from LearningPolicy import LearningPolicy

class LearningAgent(Agent):
    def __init__(self, name, mode="learning"):
        super().__init__(name)
        self.policy = LearningPolicy()
        self.mode = mode
        self.last_action = None
        self.last_observation = None

    def observe(self, observation):
        self.last_observation = observation

    def act(self):
        action_type = self.policy.choose_action(self.last_observation)
        self.last_action = action_type
        return Action(action_type)

    def evaluate(self, reward):
        if self.mode == "learning":
            self.policy.update(
                obs=self.last_observation,
                action=self.last_action,
                reward=reward,
                next_obs=self.last_observation
            )
