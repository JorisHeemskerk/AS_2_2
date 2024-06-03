from basePolicy import BasePolicy
from baseMaze import BaseMaze


class BaseAgent:
    """
    BaseAgent class.

    This agents implements a given `Policy`, which it uses to act.
    The agent does not carry out any other logic.
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
        """
        
        self.maze = maze
        self.policy = policy
        self.current_coordinate = start_coordinate

    def act(self, print_agent: bool=False)-> None:
        """
        Act method for agent.

        Base agent just perform the policy action, until it has found
        one that does not throw an IndexError.

        @param print_agent print agent after action, if True.
        """
        while True:
            try:
                action = self.policy.select_action(
                    self.maze[self.current_coordinate]
                )
                
                # no actions to be taken if terminal state is reached
                if action == None:
                    break
                
                self.current_coordinate = self.maze.step(
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

    def __str__(self)-> str:
        """
        Stringify current agent.
        
        Example:\n

        \nThis agent is currently standing at (1, 3), 
        with policy BasePolicy, in the following maze:
        \nMaze class, with following grid:
        \n┌──────────────────┬──────────────────┬──────────────────┐
        \n│ ( 0,3 ), r =  0  │ ( 1,3 ), r = 90  │ ( 2,3 ), r = 22  │
        \n├──────────────────┼──────────────────┼──────────────────┤
        \n│ ( 0,2 ), r = 57  │ ( 1,2 ), r = 28  │ ( 2,2 ), r = 35  │
        \n├──────────────────┼──────────────────┼──────────────────┤
        \n│ ( 0,1 ), r = 25  │ ( 1,1 ), r = 23  │ ( 2,1 ), r = 91  │
        \n├──────────────────┼──────────────────┼──────────────────┤
        \n│ ( 0,1 ), r = 25  │ ( 1,1 ), r = 23  │ ( 2,1 ), r = 91  │
        \n└──────────────────┴──────────────────┴──────────────────┘

        @return str with stringified current agent
        """
        return f"This agent is currently standing at "\
            f"\033[1m{self.current_coordinate}\033[0m, "\
            f"with policy \033[1m{self.policy}\033[0m, "\
            f"in the following maze:\n"\
            f"{self.maze.__str__(self.current_coordinate)}"
