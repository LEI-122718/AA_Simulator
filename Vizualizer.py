import numpy as np
import os
import time

class Vizualizer:

    def __init__(self, environment, agents):
        self.env = environment
        self.agents = agents

    def render(self):
        # limpa terminal
        os.system('cls' if os.name == 'nt' else 'clear')

        grid = self.env.grid.copy()

        # marcar agentes
        for agent, pos in self.env.agent_positions.items():
            x, y = pos
            grid[y, x] = -1   # código do agente


        # imprimir
        for row in grid:
            line = ""
            for val in row:
                if val == -1:
                    line += " A "    # agente
                elif val == 9:
                    line += " F "    # farol
                elif val > 0:
                    line += " R "    # recurso
                else:
                    line += " . "    # vazio
            print(line)
        # imprimir sensores logo depois do mapa
        print("\n--- Sensores dos agentes ---")
        for agent in self.agents:
            if agent.last_observation:
                print(f"{agent.name}: {agent.last_observation.info}")


    def update(self, delay=0.2):
        self.render()
        time.sleep(delay)
