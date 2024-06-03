from typing import Annotated
import ASCII_table

from basePolicy import BasePolicy
from baseMaze import BaseMaze
from baseAgent import BaseAgent
from floatRange import FloatRange, check_annotated
from helper import state_dict_to_np_matrix


class TemporalDifferenceAgent(BaseAgent):
    """
    TemporalDifferenceAgent class.
    """

    def __init__(
        self, 
        maze: BaseMaze, 
        policy: BasePolicy, 
        start_coordinate: tuple[int, int]
    )-> None:
        """
        @var $maze
        **Maze** `Maze` in which the agent is present.
        @var $policy 
        **Policy** `Policy` which the agent uses to act
        @var $current_coordinate 
        **tuple[int, int]** Current x, y coord of agent.
        @var $values
        **dict[State : float]** values per state in dict.
        """
        
        self.maze = maze
        self.policy = policy
        self.current_coordinate = start_coordinate
        self.values = {}

    def act(self, print_agent: bool=False)-> float:
        """
        Act method for agent.

        Base agent just perform the policy action, until it has found
        one that does not throw an IndexError.

        @param print_agent print agent after action, if True.

        @return float with reward for action
        """
        while True:
            try:
                action = self.policy.select_action(
                    self.maze[self.current_coordinate]
                )
                
                # no actions to be taken if terminal state is reached
                if action == None:
                    break
                
                self.current_coordinate, reward = self.maze.step_reward(
                    self.current_coordinate, 
                    action
                )
                break
            except IndexError:
                continue
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
        if print_agent:
            print(self)
        return reward
    
    @check_annotated
    def temporal_difference(
        self, 
        alpha: Annotated[float, FloatRange(0.0, 1.0)]=0.1,
        gamma: Annotated[float, FloatRange(0.0, 1.0)]=0.1,
        print_agent: bool=False,
        print_result: bool=False
    )-> None:
        """
        Temporal difference function for TemporalDifferenceAgent.

        This function performs the Temporal Difference algorithm.

        @param alpha: alpha from formula, idk what it does exactly.
        @param gamma: discount value
        @param print_agent: whether or not to print each step taken
        @param print_result: whether to print the final values
        """
        # Save starting point to reset at the end of the episode
        starting_coordinate = self.current_coordinate

        current_state = self.maze[self.current_coordinate]
        while not current_state.is_terminal:
            # initialise V(s) if s has not been visited before
            if current_state not in self.values:
                self.values[current_state] = 0
            # initialise V(s') if s has not been visited before
            reward = self.act(print_agent)
            resulting_state = self.maze[self.current_coordinate]
            if resulting_state not in self.values:
                self.values[resulting_state] = 0

            # Calculate V(s)
            self.values[current_state] += alpha * (
                reward + 
                (gamma * self.values[resulting_state]) - 
                self.values[current_state]
            )
            
            current_state = resulting_state

        if print_result:
            print(f"\033[32m{'─'*45}\n\t\tValue matrix\n{'─'*45}\033[0m")
            table = ASCII_table.ASCIITable(
                state_dict_to_np_matrix(self.values, '', "V = ").T[::-1]
            )
            table.print()
        self.current_coordinate = starting_coordinate
