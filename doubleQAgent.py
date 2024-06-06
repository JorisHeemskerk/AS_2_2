import copy
from typing import Annotated
import ASCII_table
import numpy as np
import random

from action import Action
from baseMaze import BaseMaze
from QAgent import QAgent
from floatRange import FloatRange, check_annotated
from helper import Q_to_np_matrix
from state import State


class DoubleQAgent(QAgent):
    """
    Double Q-learning agent.

    Extends QAgent class
    @see QAgent.py
    """

    def __init__(
        self, 
        maze: BaseMaze, 
        start_coordinate: tuple[int, int]
    )-> None:
        """
        @var $maze
        **Maze** `Maze` in which the agent is present.
        @var $policy 
        **Policy** `Policy` which the agent uses to act
        @var $current_coordinate 
        **tuple[int, int]** Current x, y coord of agent.
        @var $Q
        **dict[State : dict[Action : float]]** values per state in dict.
        @var $Q_two
        **dict[State : dict[Action : float]]** values per state in dict.
        """
        super().__init__(maze, start_coordinate)
        self.Q_two: dict[State : dict[Action : float]] = {}
    
    @check_annotated
    def Q_learning(
        self, 
        alpha: Annotated[float, FloatRange(0.0, 1.0)]=0.1,
        epsilon: Annotated[float, FloatRange(0.0, 1.0)]=0.1,
        gamma: Annotated[float, FloatRange(0.0, 1.0)]=0.1,
        print_result: bool=False
    )-> None:
        """
        Double Q-learning function for QAgent.

        This function performs the double Q-learning algorithm.

        @param alpha: alpha from formula, idk what it does exactly
        @param epsilon: epsilon from formula, idk what it does exactly
        @param gamma: discount value
        @param print_result: whether to print the final values
        """
        current_state = self.maze[self.current_coordinate]
        if current_state not in self.Q:
            self.Q[current_state] = {action : 0.0 for action in Action}
            self.Q_two[current_state] = {action : 0.0 for action in Action}
        
        while not current_state.is_terminal:
            # calculate a
            action = self._choose_action(
                {
                    key: 
                    self.Q[current_state][key] + 
                    self.Q_two[current_state][key] 
                    for key in self.Q[current_state]
                },
                epsilon
            )

            # calculate s'
            state_prime = self.maze[
                self.maze.step(current_state.position, copy.copy(action))
            ]
            # calculate r
            reward = state_prime.reward
            
            # add to Q if not yet in there
            if state_prime not in self.Q:
                self.Q[state_prime] = {action : 0.0 for action in Action}
                self.Q_two[state_prime] = {action : 0.0 for action in Action}

            # choose Q1 or Q2
            q_ref = None
            if bool(random.getrandbits(1)):
                q_ref = self.Q_two
            else:
                q_ref = self.Q

            # calculate a'
            action_prime = self._choose_action(
                q_ref[state_prime],
                0
            )

            # Q(s,a) = Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
            q_ref[current_state][action] += alpha * (
                reward + 
                (gamma * q_ref[state_prime][action_prime]) - 
                q_ref[current_state][action]
            )

            # set back current state
            current_state = state_prime
            
        if print_result:
            colour_matrix = np.array([
                [
                    ASCII_table.Colours.DEFAULT, 
                    ASCII_table.Colours.DEFAULT, 
                    ASCII_table.Colours.DEFAULT, 
                    ASCII_table.Colours.RED    
                ], [
                    ASCII_table.Colours.DEFAULT, 
                    ASCII_table.Colours.DEFAULT, 
                    ASCII_table.Colours.BLUE, 
                    ASCII_table.Colours.BLUE   
                ], [
                    ASCII_table.Colours.DEFAULT, 
                    ASCII_table.Colours.DEFAULT, 
                    ASCII_table.Colours.DEFAULT, 
                    ASCII_table.Colours.DEFAULT
                ], [
                    ASCII_table.Colours.RED, 
                    ASCII_table.Colours.DEFAULT, 
                    ASCII_table.Colours.DEFAULT, 
                    ASCII_table.Colours.DEFAULT
                ],
            ])

            # Q1
            print(f"\033[32m{'─'*57}\n\t\tQ-value matrix\n{'─'*57}\033[0m")
            table = ASCII_table.ASCIITable(
                Q_to_np_matrix(self.Q, 2, 'unvisited').T[::-1],
                colour_matrix
            )
            table.print()
            
            # self.Q_two
            print(f"\033[32m{'─'*57}\n\t\tQ_two-value matrix\n{'─'*57}\033[0m")
            table = ASCII_table.ASCIITable(
                Q_to_np_matrix(self.Q, 2, 'unvisited').T[::-1],
                colour_matrix
            )
            table.print()
