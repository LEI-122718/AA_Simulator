class SimulationEngine:
    def __init__(self, environment, agents, visualizer=None):
        self.env = environment
        self.agents = agents
        self.visualizer = visualizer

    def run(self, steps=100, delay=0.1):
        import time

        def step(i=0):
            if i >= steps:
                return

            for agent in self.agents:
                obs = self.env.get_observation(agent)
                agent.observe(obs)

            for agent in self.agents:
                action = agent.act()
                self.env.apply_action(agent, action)

            self.env.update()

            if self.visualizer:
                self.visualizer.update()

            self.visualizer.window.after(int(delay * 1000), step, i + 1)

        step()
        self.visualizer.start()



