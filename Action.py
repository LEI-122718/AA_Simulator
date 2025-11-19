from enum import Enum

class ActionType(Enum):
    STAY = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    COLLECT = 5
    DROP = 6

class Action:
    def __init__(self, action_type: ActionType):
        self.type = action_type

    def __repr__(self):
        return f"Action({self.type.name})"
