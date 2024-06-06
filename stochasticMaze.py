from typing import Annotated
import numpy as np
import random

from action import Action
from stupidMaze import StupidMaze
from floatRange import FloatRange, check_annotated


class StochasticMaze(StupidMaze):
    """
    Stochastic stupid Maze class.

    Extends StupidMaze class
    @see stupidMaze.py
    """

    @check_annotated
    def __init__(
        self, 
        grid_shape: tuple[int, int], 
        rewards: np.ndarray,
        probability: Annotated[float, FloatRange(0.0, 1.0)]
    )-> None:
        """
        @var $states
        **np.ndarray** Numpy matrix with all the states.
        @var $probability
        **Annotated[float, FloatRange(0.0, 1.0)]** 
        probability to NOT perform desired action. Should be low.
        """
        super().__init__(grid_shape, rewards)
        self.probability = probability

    def step(
        self, 
        start_coordinate: tuple[int, int], 
        action: Action
    )-> tuple[int, int]:
        """
        Step function for StochasticMaze.

        Return corresponding end coordinate, for given action.
        If action would result in agent going out of bounce, 
        the start_coordinate will be returned.

        @param start_coordinate: coordinate from where the `action`
            will be performed.
        @param action: action to take at `start_coordinate`

        @return tuple[int, int] with end coordinate
        """
        # start_coordinate must be inside of maze
        try:
            self.states[start_coordinate]
        except IndexError:
            raise IndexError(
                f"`start_coordinate` out of range."
                f" Tried accessing {start_coordinate} from `self.states`, "
                f"which has shape of {self.states.shape}"
            )

        # does not completely adhere to the probability, but fuck that
        dice_roll = random.random()
        if dice_roll < self.probability:
            action = random.choice([
                Action.UP, 
                Action.DOWN, 
                Action.LEFT, 
                Action.RIGHT
            ])

        new_coordinate = tuple(map(sum, zip(start_coordinate, action.value)))

        # if `new_coordinate`` is out of bounce at the bottom or left, 
        # let agent stay in place
        if new_coordinate[0] < 0 or new_coordinate[1] < 0:
            return start_coordinate
        
        # if `new_coordinate`` is out of bounce at the top or right, 
        # let agent stay in place           
        try:
            self.states[new_coordinate]
        except IndexError:
            return start_coordinate
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

        return new_coordinate
