import random

from typing import Annotated

from baseAgent import BaseAgent
from basePolicy import BasePolicy
from floatRange import FloatRange, check_annotated
from baseMaze import BaseMaze


class ProbabilityAgent(BaseAgent):
    """
    ProbabilityAgent class.

    This agents implements a given `Policy`, which it uses to act.
    The agent takes into account probabilities for his actions.

    The probability given is the probability that the agent will perfrom
    the action it desires. The other actions equally share the left-
    over probability. 

    E.g., if all actions are possible, and the agent has a probabilty of
    0.7, the chances of the agent moving up, if it desires to do so,
    are 70%, and the chances for any of the other actions are 10%.
    """

    @check_annotated
    def __init__(
        self, 
        maze: BaseMaze, 
        policy: BasePolicy, 
        start_coordinate: tuple[int, int],
        probability: Annotated[float, FloatRange(0.0, 1.0)]=1.0
    )-> None:
        """
        @var $maze
        **Maze** `BaseMaze` in which the agent is present.
        @var $policy 
        **Policy** `Policy` which the agent uses to act
        @var $current_coordinate 
        **tuple[int, int]** Current x, y coord of agent.
        @var $probability 
        **float** Chance for agent to perform desired action.
        """
        super().__init__(maze, policy, start_coordinate)
        self.probability = probability

        
    def act(
        self, 
        print_agent: bool=False
    )-> None:
        """
        Act method for agent.

        Base agent just perform the policy action, until it has found
        one that does not throw an IndexError.

        @param print_agent print agent after action, if True.
        """
        while True:
            desired_action = None
            try:
                action = self.policy.select_action(
                    self.maze[self.current_coordinate]
                )
                desired_action = action
                # no actions to be taken if terminal state is reached
                if action == None:
                    break

                # take into account that a decision can fail.
                dice_roll = random.random()
                if dice_roll > self.probability:
                    new_choices = list(
                            self.maze.get_destinations(
                                self.maze[self.current_coordinate]
                            ).keys()
                        )
                    new_choices.remove(action)
                    action = random.choice(new_choices)
                
                self.current_coordinate = self.maze.step(
                    self.current_coordinate, 
                    action
                )
                break
            except IndexError:
                continue
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                exit(-1)
        if print_agent:
            if desired_action != action:
                print(
                    f"\033[34mOh no! Agent tried performing {desired_action}, "
                    f"but failed and performed {action} instead! :(\033[0m"
                )
            print(self)
