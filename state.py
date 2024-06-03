class State:
    """
    State class.

    This class implements a state, used in Maze.
    @see maze.py

    A state is a position in a maze, that has a location and reward.
    A state can also be terminal.
    """

    def __init__(
        self, 
        position: tuple[int, int], 
        reward: float, 
        is_terminal: bool
    )-> None:
        """
        @var $position
        **tuple[int, int]** (x,y) coordinate of State.
        @var $reward 
        **float** Reward for entering current State.
        @var $is_terminal 
        **bool** Indicator of terminal State.
        """
        self.position = position
        self.reward = reward
        self.is_terminal = is_terminal

    def __hash__(self)-> int:
      """
      Hash function for State.
      
      This function uses the static member variables 
      to create a unique hash for the class.

      NOTE: Changing any of the variables after initialization 
      will cause a different hash to be generated.
      
      @return int with hash
      """
      return hash((self.position, self.reward, self.is_terminal))

    def __eq__(self, rhs: 'State')-> bool:
        """
        == operator for State.

        All private member variables should be the same 
        for the class to be considered equal.

        @param rhs: State object to compare to lhs
        
        @return bool with true if rhs is equal to lhs
        """
        return (self.position == rhs.position) and \
            (self.reward == rhs.reward) and \
            (self.is_terminal == rhs.is_terminal)
    
    def __lt__(self, other):
        if self.position[0] == other.position[0]:
            return self.position[1] < other.position[1]
        return self.position[0] < other.position[0]

    def __str__(self, colour: str = "\033[0m") -> str:
        """
        Stringify current state.

        @param colour: terminal colour.
        
        Example:\n
        (( 0,3 ), r = 63 

        NOTE: A terminal state will be painted red.

        @return str with stringified state
        """
        line = "({: 2d},{:^2}), r = {:^3}".format(
            self.position[0], 
            self.position[1], 
            self.reward
        )

        if self.is_terminal:
            return f"\033[31m{line}\033[0m"
        else:
            return f"{colour}{line}\033[0m"
