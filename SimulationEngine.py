class SimulationEngine:
    def __init__(self, environment, agents, visualizer=None):
        self.env = environment
        self.agents = agents
        self.visualizer = visualizer

    def run(self, steps=100, delay=0.1):
        import time

        for i in range(steps):

            # Observa
            for agent in self.agents:
                obs = self.env.get_observation(agent)
                agent.observe(obs)

            # Age
            for agent in self.agents:
                action = agent.act()
                self.env.apply_action(agent, action)

            # Atualiza ambiente
            self.env.update()

            # Visualização (ASCII ou Tk)
            if self.visualizer:
                self.visualizer.update(delay)

        # Só chama start() se existir (Tkinter)
        if self.visualizer and hasattr(self.visualizer, "start"):
            self.visualizer.start()
