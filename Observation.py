class Observation:

    def __init__(self, environment, agent):
        self.local_map = {}
        self._build_map(environment, agent)

    def _build_map(self, environment, agent):
        ax, ay = agent.position

        grid = environment.grid
        w = environment.width
        h = environment.height

        sensor = agent.sensor

        for (dx, dy) in sensor.get_visible_offsets():
            # Calcular posição absoluta no mundo
            x = ax + dx
            y = ay + dy

            # 3. Verificar se está DENTRO dos limites
            if 0 <= x < w and 0 <= y < h:
                # Apenas processamos se for válido
                cell_content = grid[x][y]

                # Guardamos no mapa local a coordenada relativa e o valor processado
                self.local_map[(dx, dy)] = self._extract_feature(cell_content)

            # O "else" desaparece.
            # Se for out-of-bounds, a chave (dx, dy) simplesmente não é criada.

    def _extract_feature(self, cell_content):
        """
        Método auxiliar para extrair apenas o valor/tipo do objeto na grelha.
        """
        if cell_content is None:
            return 0  # Espaço vazio

        # Se o objeto tiver um valor específico (ex: comida=10)
        if hasattr(cell_content, 'value'):
            return cell_content.value

        return 1  # Objeto genérico/Obstáculo

    def get_map(self):
        """
        Retorna o dicionário {(dx, dy): valor}.
        Nota: Coordenadas fora do mapa não estarão presentes nas chaves.
        """
        return self.local_map

    #temos o mapa grande que é o ambiante e  a observação é um mmini mapara do enviroment o tamanhi da observação é definida pelo sendore
    #e o get observation depende o enviroment sendo implemnetaod em cada mapa.