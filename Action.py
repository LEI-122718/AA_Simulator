from enum import Enum

class ActionType(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    STAY = 5
    COLLECT = 6
    DROP = 7


class Action:
    def __init__(self, action_type):
        self.type = action_type

    # CORREÇÃO IMPORTANTE:
    # Este método não existia e Lighthouse chama act.to_vector().
    # Agora convertemos a ação num deslocamento (dx,dy).
    def to_vector(self):
        if self.type == ActionType.UP:
            return (0, -1)
        if self.type == ActionType.DOWN:
            return (0, 1)
        if self.type == ActionType.LEFT:
            return (-1, 0)
        if self.type == ActionType.RIGHT:
            return (1, 0)
        return (0, 0)  # STAY
