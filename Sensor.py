class Sensor:
    """
    Apenas define o raio de visão do agente.
    NÃO lê valores do ambiente.
    """
    def __init__(self, radius=1):
        self.radius = radius

    def get_visible_offsets(self):
        offsets = []
        for dy in range(-self.radius, self.radius + 1):
            for dx in range(-self.radius, self.radius + 1):
                offsets.append((dx, dy))
        return offsets
