from action import Action
from state import State
from basePolicy import BasePolicy


class HardcodedOptimalPolicy(BasePolicy):
    """
    HardcodedOptimalPolicy class.

    This class can be constructed with an hardcoded optimal policy
    """

    def __init__(self, actions: dict[State : Action])-> None:
        """
        Initializer for HardcodedOptimalPolicy.

        Sets self.actions.
        """
        super().__init__()

        self.actions = actions

    def select_action(self, state: State)-> Action:
        """
        Select action based on current policy.
        
        This policy works as follows:
        - select select the best action, given the MDP, and return it.

        @param state: Current State to perform action in.

        @return Action with Action to perform.
        """
        return self.actions[state]
    