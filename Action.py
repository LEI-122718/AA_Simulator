import random
from enum import Enum

class ActionType(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    STAY = 5

class Action:
    def __init__(self, action_type):
        self.type = action_type