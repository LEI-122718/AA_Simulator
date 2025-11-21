# LearningAgent.py

from Agent import Agent
from Action import Action
from LearningPolicy import LearningPolicy


class LearningAgent(Agent):
    def __init__(self, name, radius=1, mode="learning"):
        super().__init__(name, radius)
        self.policy = LearningPolicy()
        self.mode = mode  # "learning" ou "test"

        # Para Q-learning
        self.last_action = None
        self.prev_observation = None   # s_t
        self.last_observation = None   # s_{t+1}

    # ------------------------------------------------------------------ #
    # Observação
    # ------------------------------------------------------------------ #
    def observe(self, observation):
        """
        Guarda par (s_t, s_{t+1}):

        - prev_observation: estado anterior
        - last_observation: estado atual
        """
        self.prev_observation = self.last_observation
        self.last_observation = observation

    # ------------------------------------------------------------------ #
    # Ação
    # ------------------------------------------------------------------ #
    def act(self):
        """
        Política: usa a LearningPolicy baseada na última observação.
        Se ainda não tem observação, fica parado.
        """
        if self.last_observation is None:
            return Action()  # por defeito ActionType.STAY

        action_type = self.policy.choose_action(self.last_observation)
        self.last_action = action_type

        return Action(action_type)

    # ------------------------------------------------------------------ #
    # Avaliação / Aprendizagem
    # ------------------------------------------------------------------ #
    def evaluate(self, reward):
        """
        Recebe recompensa do ambiente após agir
        e atualiza Q (apenas em modo "learning").
        """
        if self.mode != "learning":
            return

        # Só atualizamos se tivermos uma transição completa
        if (
            self.prev_observation is None
            or self.last_observation is None
            or self.last_action is None
        ):
            return

        self.policy.update(
            obs=self.prev_observation,
            action=self.last_action,
            reward=reward,
            next_obs=self.last_observation,
        )

    # ------------------------------------------------------------------ #
    def communicate(self, message: str, from_agent):
        # Não usado neste projeto
        pass
