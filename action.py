from enum import Enum


class Action(Enum):
    """
    Action Enum class.

    This class makes it such that every action taken can be described
     using text instead of the numbers.
    """
    UP    = (0, 1)
    DOWN  = (0, -1)
    LEFT  = (-1, 0)
    RIGHT = (1, 0)
