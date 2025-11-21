import numpy as np
from Environment import Environment
from Observation import Observation
from Action import ActionType


class Foraging(Environment):
    """
    Ambiente de Recoleção (Foraging) com visão espacial via Observation(self, agent).

    - self.resources[y, x]  -> quantidade de recurso (>= 0) ou obstáculo (< 0, opcional)
    - self.nest_pos         -> posição do ninho
    - self.grid             -> o que os agentes "vêem" (recursos + marca do ninho)
    """

    def __init__(
        self,
        grid,
        nest_pos,
        step_penalty=-0.01,
        collect_reward=1.0,
        drop_reward=5.0,
        failed_action_penalty=-0.05,
    ):
        """
        grid: lista de listas com valores (>=0 recurso, <0 obstáculo opcional)
        nest_pos: tuplo (x, y) com posição do ninho
        """
        height = len(grid)
        width = len(grid[0])

        super().__init__(width, height)

        # Recursos "físicos" (quanto há em cada célula)
        self.resources = np.array(grid, dtype=float)

        # Posição do ninho
        self.nest_pos = np.array(nest_pos, dtype=int)

        # Grid observada pelos agentes (recursos + ninho marcado)
        self.grid = self._build_grid()

        # Estado dos agentes
        self.agent_positions = {}
        self.carrying = {}           # agent -> bool
        self.step_reward = {}        # recompensa do último passo
        self.cumulative_reward = {}  # recompensa acumulada por episódio

        # Parâmetros de reward
        self.step_penalty = step_penalty
        self.collect_reward = collect_reward
        self.drop_reward = drop_reward
        self.failed_action_penalty = failed_action_penalty

    # ------------------------------------------------------------------
    # Grid observada (recursos + ninho)
    # ------------------------------------------------------------------

    def _build_grid(self):
        """
        Constrói a grid que será usada pelo Observation:
        - começa pelos recursos
        - marca o ninho com um valor alto (sem destruir recurso eventualmente existente)
        """
        grid = self.resources.copy()

        nx, ny = int(self.nest_pos[0]), int(self.nest_pos[1])
        if 0 <= nx < self.width and 0 <= ny < self.height:
            # garantir que o ninho é claramente visível (>= 9)
            grid[ny, nx] = max(grid[ny, nx], 9.0)

        return grid

    # ------------------------------------------------------------------
    # Gestão de agentes
    # ------------------------------------------------------------------

    def add_agent(self, agent, pos):
        """
        Adiciona agente ao ambiente numa posição (x, y).
        """
        self.agent_positions[agent] = np.array(pos, dtype=int)
        self.carrying[agent] = False
        self.step_reward[agent] = 0.0
        self.cumulative_reward[agent] = 0.0

    # ------------------------------------------------------------------
    # Observação (mini-mapa via sensor, igual ao Lighthouse)
    # ------------------------------------------------------------------

    def get_observation(self, agent):
        """
        Usa Observation(self, agent), que constrói o mini-mapa com base em:
        - environment.grid
        - environment.agent_positions
        - agent.sensor (raio de visão)
        """
        return Observation(self, agent)

    # ------------------------------------------------------------------
    # Dinâmica das ações
    # ------------------------------------------------------------------

    def apply_action(self, agent, action):
        """
        Ações suportadas:
          - UP, DOWN, LEFT, RIGHT: movimento (com limites e obstáculos opcionais)
          - STAY: nada
          - COLLECT: tenta recolher recurso na célula atual
          - DROP: tenta entregar recurso no ninho

        A recompensa deste passo é calculada aqui e guardada em step_reward[agent],
        e também acumulada em cumulative_reward[agent].
        """
        # reset da recompensa deste passo
        self.step_reward[agent] = 0.0

        # posição atual
        x, y = self.agent_positions[agent]
        orig_x, orig_y = x, y

        # ------------------------ Movimento ------------------------
        dx, dy = action.to_vector()
        x += dx
        y += dy
        # COLLECT, DROP não mexem posição por si (dx, dy = 0, 0)

        # aplica limites do mapa
        x = int(np.clip(x, 0, self.width - 1))
        y = int(np.clip(y, 0, self.height - 1))

        # Se quiseres obstáculos: qualquer célula com resources < 0 é intransponível
        if self.resources[y, x] < 0:
            # cancela movimento e volta à posição original
            x, y = orig_x, orig_y

        # atualiza posição
        self.agent_positions[agent] = np.array([x, y])

        # penalização por passo (mesmo se não fez nada útil)
        self.step_reward[agent] += self.step_penalty

        # estado depois do movimento
        cell_value = self.resources[y, x]
        at_nest = (x == self.nest_pos[0] and y == self.nest_pos[1])

        # ------------------------ COLLECT ------------------------
        if action.type == ActionType.COLLECT:
            if (cell_value > 0) and (not self.carrying[agent]):
                # recolhe 1 unidade de recurso
                self.resources[y, x] -= 1
                self.carrying[agent] = True
                self.step_reward[agent] += self.collect_reward
            else:
                # tentativa falhada (sem recurso ou já a carregar)
                self.step_reward[agent] += self.failed_action_penalty

        # ------------------------ DROP ------------------------
        if action.type == ActionType.DROP:
            if self.carrying[agent] and at_nest:
                self.carrying[agent] = False
                self.step_reward[agent] += self.drop_reward
            else:
                # tentativa falhada
                self.step_reward[agent] += self.failed_action_penalty

        # acumular total do episódio
        self.cumulative_reward[agent] += self.step_reward[agent]

    # ------------------------------------------------------------------
    # Atualização do ambiente
    # ------------------------------------------------------------------

    def update(self):
        """
        Atualiza o ambiente após cada passo.
        Aqui voltamos a construir a grid observada (recursos + ninho).
        Se quiseres dinâmica extra (regenerar recursos, etc.), é aqui.
        """
        self.grid = self._build_grid()

    # ------------------------------------------------------------------
    # Recompensa para RL
    # ------------------------------------------------------------------

    def compute_reward(self, agent):
        """
        Devolve a recompensa do ÚLTIMO passo para o agente dado.
        É isto que o Q-learning deve usar.
        """
        return float(self.step_reward.get(agent, 0.0))
