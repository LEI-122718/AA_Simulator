from Foraging import Foraging
from Lighthouse import Lighthouse
from SimpleAgent import SimpleAgent
from SimulationEngine import SimulationEngine
from Vizualizer import Vizualizer


env = Lighthouse(10, 10, lighthouse_pos=(6, 7))
agent = SimpleAgent("A1")

env.add_agent(agent, pos=(2, 2))

vis = Vizualizer(env, [agent])


sim = SimulationEngine(env, [agent], vis)
sim.run(steps=30, delay=0.2)


