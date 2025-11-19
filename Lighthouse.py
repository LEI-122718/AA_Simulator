import numpy as np
from Environment import Environment
from Observation import Observation

class Lighthouse(Environment):
    def __init__(self, width, height, lighthouse_pos):
        super().__init__()

        # Representação do mapa (0 = vazio)
        self.grid = np.zeros((height, width), dtype=int)

        # 9 = farol
        lx, ly = lighthouse_pos
        self.grid[ly, lx] = 9

        self.lighthouse_pos = lighthouse_pos
        self.agent_positions = {}

        self.height = height
        self.width = width

    def add_agent(self, agent, pos):
        self.agent_positions[agent] = np.array(pos, dtype=int)

    def get_observation(self, agent):
        ax, ay = self.agent_positions[agent]
        lx, ly = self.lighthouse_pos

        dx = lx - ax
        dy = ly - ay

        if abs(dx) > abs(dy):
            direction = "RIGHT" if dx > 0 else "LEFT"
        else:
            direction = "DOWN" if dy > 0 else "UP"

        return Observation({"direction": direction})

    def apply_action(self, agent, action):
        x, y = self.agent_positions[agent]

        if action.type.name == "UP":    y -= 1
        if action.type.name == "DOWN":  y += 1
        if action.type.name == "LEFT":  x -= 1
        if action.type.name == "RIGHT": x += 1

        y = np.clip(y, 0, self.grid.shape[0]-1)
        x = np.clip(x, 0, self.grid.shape[1]-1)

        self.agent_positions[agent] = np.array([x, y], dtype=int)

    def update(self):
        pass
