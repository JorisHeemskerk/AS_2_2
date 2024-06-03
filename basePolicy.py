import random

from action import Action
from baseMaze import BaseMaze
from state import State


class BasePolicy:
    """
    BasePolicy
    
    Base policy class with random behavior.
    This policy works as follows:
    - select select a random action and return it.
    """

    def __init__(self)-> None:
        """
        Initializer for BasePolicy.

        This class has no member variables, 
        meaning the initializer does nothing.
        """
        pass

    def select_action(self, state: State)-> Action:
        """
        Select action based on current policy.
        
        This policy works as follows:
        - select select a random action and return it.

        @param state: Current State to perform action in.

        @return Action with Action to perform.
        """
        index = random.randrange(0, 4)
        return [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT][index]
    
    def visualise(self, maze: BaseMaze)-> None:
        """
        print current Policy.
        
        Example:\n
        
        \n┌────────────────┬────────────────┬────────────────┐
        \n│ ( 0,3 ), a = ► │ ( 1,3 ), a = ► │ ( 2,3 ), a = ► │
        \n├────────────────┼────────────────┼────────────────┤
        \n│ ( 0,2 ), a = ▼ │ ( 1,2 ), a = ▲ │ ( 2,2 ), a = ▲ │
        \n├────────────────┼────────────────┼────────────────┤
        \n│ ( 0,1 ), a = ▼ │ ( 1,1 ), a = ◄ │ ( 2,1 ), a = ► │
        \n├────────────────┼────────────────┼────────────────┤
        \n│ ( 0,0 ), a = X │ ( 1,0 ), a = ◄ │ ( 2,0 ), a = ◄ │
        \n└────────────────┴────────────────┴────────────────┘

        @param maze: BaseMaze object to visualise policy in.
        """

        action_to_arrow = {
            Action.UP : "▲",
            Action.DOWN : "▼",
            Action.LEFT : "◄",
            Action.RIGHT : "►",
            None: "✕"
        }

        # base case for the horizontal lines
        deviding_line = \
            f"{('─' * 16 + '┼') * (maze.states.shape[0] - 1)}"\
            f"{'─' * 16}"

        output = f"┌{deviding_line.replace('┼', '┬')}┐\n│ "

        # transform and reverse matrix, 
        # such that (0, 0) starts in the bottom left
        reversed_transformed_states = maze.states.T[::-1]
        for row in reversed_transformed_states[:-1]:
            for state in row.tolist():
                output += str(state).split('r')[0] + \
                    "\033[35ma = {:^1}\033[0m │ ".format(
                        action_to_arrow[self.select_action(state)]
                    )
            output += f"\n├{deviding_line}┤\n│ "

        # different formatting for last line
        for state in reversed_transformed_states[-1].tolist():
            output += str(state).split('r')[0] + \
                "\033[35ma = {:^1}\033[0m │ ".format(
                        action_to_arrow[self.select_action(state)]
                    )
        output += f"\n└{deviding_line.replace('┼', '┴')}┘"
        print(output)
    
    def __str__(self) -> str:
        """
        stringify policy by just returning the class name.

        @return str with class name
        """
        return self.__class__.__name__
