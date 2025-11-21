class Observation:

    def __init__(self, environment, agent):
        self.local_map = {}
        self.build_map(environment, agent)
        self.info = self._build_info()
    # ----------------------------------------------------------------------

    def build_map(self, environment, agent):
        """
        Constrói o mini-mapa com base no sensor do agente.
        """

        # CORREÇÃO:
        # Antes usava agent.position, mas Lighthouse guarda posições no dicionário.
        ax, ay = environment.agent_positions[agent]

        grid = environment.grid  # NumPy array
        H, W = grid.shape

        sensor = agent.sensor

        for (dx, dy) in sensor.get_visible_offsets():

            # Posição absoluta no mapa grande
            x = ax + dx
            y = ay + dy

            # Verificar limites
            if 0 <= x < W and 0 <= y < H:

                # CORREÇÃO:
                # Antes: grid[x][y] → invertido.
                value = grid[y, x]

                self.local_map[(dx, dy)] = value

            # Se sair fora do mapa, simplesmente não adicionamos a chave
            # Isto evita erros.

    # ----------------------------------------------------------------------
    def _build_info(self):
        """
        Cria um estado compacto baseado no mini-mapa.
        LearningPolicy precisa de observation.info como dict.
        """

        # Convertemos o mini-mapa em pares ((dx,dy), valor)
        # Isto produz um estado estável e distinguível.
        info = {}

        for key, value in self.local_map.items():
            info[key] = float(value)

        return info
    def get_map(self):
        """Retorna o mini-mapa como dicionário {(dx,dy): valor}."""
        return self.local_map
