import numpy as np

from action import Action
from baseMaze import BaseMaze


class StupidMaze(BaseMaze):
    """
    Maze class that lets agent perform every possible action.

    Extends BaseMaze class
    @see baseMaze.py
    """

    def __init__(
        self, 
        grid_shape: tuple[int, int], 
        rewards: np.ndarray
    )-> None:
        """
        @var $states
        **np.ndarray** Numpy matrix with all the states.
        """
        super().__init__(grid_shape, rewards)

    def step(
        self, 
        start_coordinate: tuple[int, int], 
        action: Action
    )-> tuple[int, int]:
        """
        Step function for StupidMaze.

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

    def step_reward(
        self, 
        start_coordinate: tuple[int, int], 
        action: Action
    )-> tuple[tuple[int, int], float]:
        """
        Step function for StupidMaze.

        Return corresponding end coordinate, for given action, 
        along with the reward.
        If action would result in agent going out of bounce, 
        the start_coordinate will be returned.

        @param start_coordinate: coordinate from where the `action`
            will be performed.
        @param action: action to take at `start_coordinate`

        @return tuple[tuple[int, int], float] 
            with end coordinate and reward
        """
        new_coordinate = self.step(start_coordinate, action)
        return new_coordinate, self[new_coordinate].reward
