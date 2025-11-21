import numpy as np
from Environment import Environment


class Lighthouse(Environment):
    """
    Ambiente do Farol.

    - self.world: mapa físico
        0    -> espaço livre
        -1   -> obstáculo simples
        -5   -> parede dura
        -10  -> proibição total
        >0   -> célula do farol (valor base da luz)

    - A partir da posição do farol, gera-se um GRADIENTE de luz:
        light[y, x] = max(0, lighthouse_value - dist_manhattan)

    - O agente observa:
        grid = light + penalizações_físicas + penalização_novidade
    """

    def __init__(
        self,
        grid,
        novelty_penalty=0.2,
        step_penalty=-0.05,
        wall_penalty=-1.0,
        goal_reward=10.0,
    ):
        height = len(grid)
        width = len(grid[0])
        super().__init__(width, height)
        # Tracking para recompensa baseada em aproximação
        self.last_distance = {}
        self.last_move_valid = {}


        # Mapa físico (obstáculos + farol)
        self.world = np.array(grid, dtype=float)

        # Campo de luz
        self.light = np.zeros((height, width), dtype=float)

        # Penalização por novidade
        self.novelty_penalty = novelty_penalty
        self.visits = np.zeros((height, width), dtype=float)

        # Posições dos agentes
        self.agent_positions = {}

        # Recompensas (mesma lógica do Foraging)
        self.step_reward = {}        # recompensa deste passo
        self.cumulative_reward = {}  # soma ao longo do episódio
        self.step_penalty = step_penalty
        self.wall_penalty = wall_penalty
        self.goal_reward = goal_reward

        # Farol
        self.lighthouse_pos, self.lighthouse_value = self._find_lighthouse()

        # Gradiente de luz
        self._compute_light_gradient()

        # Grid final observada
        self.grid = self._compute_total_grid()

    # ------------------------------------------------------------------

    def _find_lighthouse(self):
        """Encontra a célula com maior valor positivo (farol)."""
        max_val = -np.inf
        pos = (None, None)

        h, w = self.world.shape
        for y in range(h):
            for x in range(w):
                if self.world[y, x] > max_val:
                    max_val = self.world[y, x]
                    pos = (x, y)

        return pos, max_val

    # ------------------------------------------------------------------

    def _compute_light_gradient(self):
        """Gera o gradiente de luz a partir do farol."""
        lx, ly = self.lighthouse_pos
        h, w = self.world.shape

        for y in range(h):
            for x in range(w):
                dist = abs(x - lx) + abs(y - ly)
                self.light[y, x] = max(0.0, self.lighthouse_value - dist)

    # ------------------------------------------------------------------

    def _compute_total_grid(self):
        """
        Combina:
          - campo de luz (light)
          - penalizações fixas (obstáculos da world)
          - penalização por novidade (visits)
        """
        physical_penalties = np.where(self.world < 0, self.world, 0.0)
        novelty_penalties = -self.novelty_penalty * self.visits
        total = self.light + physical_penalties + novelty_penalties
        return total

    # ------------------------------------------------------------------

    def add_agent(self, agent, pos):
        """Regista o agente e inicializa recompensas."""
        self.agent_positions[agent] = np.array(pos, dtype=int)
        self.step_reward[agent] = 0.0
        self.cumulative_reward[agent] = 0.0

        # inicializa distância ao farol
        fx, fy = self.lighthouse_pos
        x, y = pos
        self.last_distance[agent] = abs(x - fx) + abs(y - fy)
        self.last_move_valid[agent] = True

    # ------------------------------------------------------------------

    def apply_action(self, agent, act):
        dx, dy = act.to_vector()
        x, y = self.agent_positions[agent]
        nx, ny = x + dx, y + dy

        # assume movimento inválido
        self.last_move_valid[agent] = False

        # movimento fora do mapa
        if not (0 <= nx < self.width and 0 <= ny < self.height):
            return

        # parede / obstáculo (qualquer valor negativo na world)
        if self.world[ny, nx] < 0:
            return

        # movimento válido → atualiza posição
        self.agent_positions[agent] = np.array((nx, ny))
        self.last_move_valid[agent] = True

    # ------------------------------------------------------------------

    def update(self):
        """
        Depois de todos agirem:
        - incrementa visitas na célula atual de cada agente
        - recalcula grid observada
        """
        for agent, pos in self.agent_positions.items():
            x, y = pos
            self.visits[y, x] += 1.0

        self.grid = self._compute_total_grid()

    # ------------------------------------------------------------------
    # Recompensa para RL
    # ------------------------------------------------------------------

    def compute_reward(self, agent):
        x, y = self.agent_positions[agent]
        fx, fy = self.lighthouse_pos

        dist_new = abs(x - fx) + abs(y - fy)
        dist_old = self.last_distance.get(agent, dist_new)

        # penalização base
        reward = self.step_penalty

        # Se bateu em parede / proibido
        if not self.last_move_valid.get(agent, True):
            reward += self.wall_penalty

        else:
            # recompensa por aproximar-se
            reward += 2.0 * (dist_old - dist_new)

            # bónus se chegou ao farol
            if (x, y) == (fx, fy):
                reward += self.goal_reward

        # guarda última distância para o próximo passo
        self.last_distance[agent] = dist_new

        # guarda também para registro interno
        self.step_reward[agent] = reward
        self.cumulative_reward[agent] += reward

        return reward

