import copy
from typing import Annotated
import ASCII_table
import numpy as np

from action import Action
from baseMaze import BaseMaze
from SARSAAgent import SARSAAgent
from floatRange import FloatRange, check_annotated
from helper import Q_to_np_matrix
from state import State


class QAgent(SARSAAgent):
    """
    Q-learning agent.

    This is SARSAMAX.

    Extends SARSAAgent class
    @see SARSAAgent.py
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
        """
        super().__init__(maze, start_coordinate)
    
    @check_annotated
    def Q_learning(
        self, 
        alpha: Annotated[float, FloatRange(0.0, 1.0)]=0.1,
        epsilon: Annotated[float, FloatRange(0.0, 1.0)]=0.1,
        gamma: Annotated[float, FloatRange(0.0, 1.0)]=0.1,
        print_result: bool=False
    )-> None:
        """
        Q-learning function for QAgent.

        This function performs the Q-learning algorithm.

        @param alpha: alpha from formula, idk what it does exactly
        @param epsilon: epsilon from formula, idk what it does exactly
        @param gamma: discount value
        @param print_agent: whether or not to print each step taken
        @param print_result: whether to print the final values
        """
        current_state = self.maze[self.current_coordinate]
        if current_state not in self.Q:
            self.Q[current_state] = {action : 0.0 for action in Action}
        
        while not current_state.is_terminal:
            # calculate a
            action = self._choose_action(
                self.Q[current_state], 
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

            # calculate a'
            
            action_prime = self._choose_action(
                self.Q[state_prime],
                0
            )

            # Q(s,a) = Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
            self.Q[current_state][action] += alpha * (
                reward + 
                (gamma * self.Q[state_prime][action_prime]) - 
                self.Q[current_state][action]
            )

            # set back current state
            current_state = state_prime
            
        if print_result:
            print(f"\033[32m{'─'*57}\n\t\tQ-value matrix\n{'─'*57}\033[0m")
            table = ASCII_table.ASCIITable(
                Q_to_np_matrix(self.Q, 2, 'unvisited').T[::-1],
                np.array([
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
            )
            table.print()
