from Agent import Agent
from Action import Action, ActionType

class SimpleAgent(Agent):
    def __init__(self, name, radius=1):
        super().__init__(name, radius)

    def observe(self, observation):
        self.last_observation = observation

    def act(self):
        """Comportamento totalmente reativo baseado no mini-mapa local."""
        if self.last_observation is None:
            return Action(ActionType.STAY)

        local_map = self.last_observation.get_map()

        # Movimentos ortogonais possíveis (no referencial local do agente)
        directions = {
            (0, -1): ActionType.UP,
            (0, 1): ActionType.DOWN,
            (-1, 0): ActionType.LEFT,
            (1, 0): ActionType.RIGHT,
        }

        best_dir = None
        best_value = -1e9

        for (dx, dy), action_type in directions.items():
            if (dx, dy) not in local_map:
                continue

            val = local_map[(dx, dy)]

            # Se for obstáculo ou extremamente penalizado, ignora
            if val < -5:
                continue

            if val > best_value:
                best_value = val
                best_dir = (dx, dy)

        # Se não encontrou nada melhor que zero, fica parado
        if best_dir is None or best_value <= 0:
            return Action(ActionType.STAY)

        # Caso contrário, retorna a ação correspondente
        return Action(directions[best_dir])
