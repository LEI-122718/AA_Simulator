import numpy as np

from Agent import Agent
from Foraging import Foraging

grid = np.zeros((10,10))

agent = Agent("A1", radius=1)
env = Foraging(grid)

env.add_agent(agent, (5,5))

obs = env.observation_for(agent)

print(obs.get_map())



