import random
from Action import ActionType


class LearningPolicy:
    """
    Política de Q-learning simples.

    - Estado é APENAS a direção de maior valor no mini-mapa local:
        ("UP"), ("DOWN"), ("LEFT"), ("RIGHT") ou ("STAY",)

      Isto reduz brutalmente o espaço de estados e permite aprender.
    """

    def __init__(self, actions=None, alpha=0.2, gamma=0.95, epsilon=0.1):
        if actions is None:
            actions = list(ActionType)

        self.actions = actions
        self.q = {}  # {(state): {action: value}}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    # ------------------------------------------------------------------ #
    # Construção de estado compacto
    # ------------------------------------------------------------------ #
    def get_state_key(self, observation):
        """
        Constrói um estado simples a partir de observation.info.

        observation.info: dict {(dx,dy): valor}

        Vamos olhar só para:
            centro (0,0)
            cima   (0,-1)
            baixo  (0, 1)
            esquerda (-1,0)
            direita  (1,0)

        O estado é um tuple com a direção do vizinho de maior valor,
        ou ("STAY",) se o melhor vizinho não for melhor que o centro.
        """
        local = observation.info

        center = local.get((0, 0), 0.0)
        neighbors = {
            "UP":    local.get((0, -1), -999.0),
            "DOWN":  local.get((0,  1), -999.0),
            "LEFT":  local.get((-1, 0), -999.0),
            "RIGHT": local.get((1,  0), -999.0),
        }

        best_dir = max(neighbors, key=neighbors.get)
        best_val = neighbors[best_dir]

        # se nada for melhor que o centro, consideramos estado "STAY"
        if best_val <= center:
            return ("STAY",)

        return (best_dir,)

    # ------------------------------------------------------------------ #
    # Escolha de ação (ε-greedy)
    # ------------------------------------------------------------------ #
    def choose_action(self, observation):
        state = self.get_state_key(observation)

        # exploração
        if random.random() < self.epsilon or state not in self.q:
            return random.choice(self.actions)

        # exploração-greedy: melhor ação segundo Q
        return max(self.q[state], key=self.q[state].get)

    # ------------------------------------------------------------------ #
    # Atualização Q-learning
    # ------------------------------------------------------------------ #
    def update(self, obs, action, reward, next_obs):
        state = self.get_state_key(obs)
        next_state = self.get_state_key(next_obs)

        if state not in self.q:
            self.q[state] = {a: 0.0 for a in self.actions}
        if next_state not in self.q:
            self.q[next_state] = {a: 0.0 for a in self.actions}

        q_old = self.q[state][action]
        q_max_next = max(self.q[next_state].values())

        # Q(s,a) <- Q(s,a) + α [ r + γ max_a' Q(s',a') - Q(s,a) ]
        self.q[state][action] = q_old + self.alpha * (
            reward + self.gamma * q_max_next - q_old
        )
