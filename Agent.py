class Agent:


    def __init__(self,x,y, name,ambient):
        self.x = x
        self.y = y
        self.name = name
        self.ambient=ambient
        self.sensors = []
        self.current_observation = None
        self.current_action = None
        self.current_reward = 0

    def createAgent(self):
        pass


    def observation(self, observation):
        self.current_observation=observation

    def takeAction(self, action):
        self.current_action=action

    def evaluateCurrentState(self,reward):
        pass

    def install(self, sensor):
        self.sensors.append(sensor)

    def comunicate(self, message, agent):
        print("Agente ", self.id_agent, " a comunicar com agente ", agent.id_agent, ": ", message)

