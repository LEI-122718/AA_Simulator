# SimulationEngine.py

class SimulationEngine:

    def __init__(self, environment, agents, visualizer=None):
        self.env = environment
        self.agents = agents
        self.visualizer = visualizer

    def run(self, steps=100, delay=0.1):
        import time

        for step in range(steps):

            # 1) Observação do estado atual
            for agent in self.agents:
                if hasattr(self.env, "get_observation") and hasattr(agent, "observe"):
                    obs = self.env.get_observation(agent)
                    agent.observe(obs)

            # 2) Agentes escolhem ação e agem
            for agent in self.agents:
                if hasattr(agent, "act"):
                    action = agent.act()
                    self.env.apply_action(agent, action)

            # 3) Ambiente atualiza (ex.: visitas no Lighthouse, dinâmica no Foraging)
            if hasattr(self.env, "update"):
                self.env.update()

            # 4) Nova observação + recompensa + aprendizagem (DEPOIS de agir)
            for agent in self.agents:
                # nova observação (estado seguinte)
                if hasattr(self.env, "get_observation") and hasattr(agent, "observe"):
                    new_obs = self.env.get_observation(agent)
                    agent.observe(new_obs)

                # recompensa (se o ambiente definir)
                reward = 0.0
                if hasattr(self.env, "compute_reward"):
                    reward = self.env.compute_reward(agent)

                # aprendizagem (se o agente suportar evaluate)
                if hasattr(agent, "evaluate"):
                    agent.evaluate(reward)

            # 5) Visualização
            if self.visualizer:
                self.visualizer.update(delay)

        # No fim, se o visualizer tiver loop próprio (ex.: Tkinter)
        if self.visualizer and hasattr(self.visualizer, "start"):
            self.visualizer.start()
