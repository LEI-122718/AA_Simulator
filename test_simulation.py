import numpy as np
from Lighthouse import Lighthouse
from SimpleAgent import SimpleAgent
from LearningAgent import LearningAgent
from LearningPolicy import LearningPolicy
from SimulationEngine import SimulationEngine
from Sensor import Sensor

print("\n=== TESTE FAROL: REATIVO vs RL ===\n")

# -------- MAPA COM GRADIENTE + OBSTÁCULOS (EXEMPLO) --------
grid = [
    [4, 5, 6, -1, -1, 5, 4],
    [4, -1, 7, 8, 7, 6, 5],
    [5, -1, 8, 9, 8, -1, 6],
    [6, 7, -1, 10, 9, -1, 7],  # farol (valor mais alto)
    [5, 5, -1, 9, 8, 7, 6]
]

lighthouse_pos = (3, 3)

# Ambiente
env = Lighthouse(grid=grid, lighthouse_pos=lighthouse_pos)

# Sensores
sensor = Sensor(radius=1)   # mini-mapa 3x3

# --------- AGENTES ---------

reactive_agent = SimpleAgent(sensor)
rl_agent = LearningAgent(policy=LearningPolicy(), sensor=sensor)

env.add_agent(reactive_agent, (0, 0))
env.add_agent(rl_agent, (6, 0))

engine = SimulationEngine(env, [reactive_agent, rl_agent])

# --------- SIMULAÇÃO ---------
STEPS = 40

for step in range(STEPS):
    print(f"\n===== STEP {step} =====")

    for agent in [reactive_agent, rl_agent]:
        pos = env.agent_positions[agent]
        print(f"{agent.__class__.__name__} posição: {pos}")

        obs = env.get_observation(agent)
        print("\n--- Mini-mapa observado ---")
        obs.pretty_print()

    engine.run_step()
