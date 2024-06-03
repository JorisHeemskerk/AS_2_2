from typing import Annotated

from action import Action
from basePolicy import BasePolicy
from floatRange import FloatRange, check_annotated
from baseMaze import BaseMaze
from state import State
        

class OptimalPolicy(BasePolicy):
    """
    OptimalPolicy
    
    Base policy class with optimal behavior, given an MDP.
    This policy works as follows:
    - select select the best action, given the MDP, and return it.
    """

    @check_annotated
    def __init__(
        self, 
        maze: BaseMaze, 
        threshold: Annotated[float, FloatRange(0.0, float("inf"))],
        discount: Annotated[float, FloatRange(0.0, 1.0)],
        probability: Annotated[float, FloatRange(0.0, 1.0)]=1.0,
        visualise: bool=False
    )-> None:
        """
        @var $maze
        **Maze** with MDP information

        @var $actions
        **dict[state : action]** 
        dictionary with states mapping to optimal actions.  
        """
        super().__init__()

        self.maze = maze
        self.actions = self._determine_optimal_policy(
            self._value_iteration(
                threshold, 
                discount,
                probability, 
                visualise
            ), 
            discount,
            probability
        )
        if visualise:
            print(f"\033[32m{'─'*47}\n\t\tOptimal Policy:\n{'─'*47}\033[0m")
            self.visualise(self.maze)

    @check_annotated
    def _value_iteration(
        self, 
        threshold: Annotated[float, FloatRange(0.0, float("inf"))],
        discount: Annotated[float, FloatRange(0.0, 1.0)],
        probability: Annotated[float, FloatRange(0.0, 1.0)]=1.0,
        visualise: bool=False
    )-> dict[State : float]:
        """
        Value iteration

        Perform bellman equation on MDP, given provided parameters,
        in order to calculate each state's value.
        
        @param threshold: float greater than 0.0 with threshold for
        when to stop converging
        @param discount: discount for future values/states
        @param probability: probability for any given action to succeed
        @param visualise: print value matrix after each iteration
        if true

        @return dict[State : float] with optimal policy
        """
        previous_values = {state: 0 for state in self.maze.states.flatten()}
        delta = float("inf")
        iteration = 0
        
        while delta >= threshold:
            delta = 0
            new_values = previous_values.copy()
            for state in self.maze.states.flatten():
                # terminal states have a value of 0 
                # and policy of no action
                if state.is_terminal:
                    new_values[state] = 0
                    continue

                # determine new value using $V(s) \leftarrow 
                # {max}_a \sum_{s',r}^{} 
                # p(s', r | s, a) [r + \gamma V(s')]$
                values_all_actions = []
                destination_states = self.maze.get_destinations(
                    state
                ).values()
                for destination_state in destination_states:
                # P * (r + \gamma * V(destination_state)) + sum(
                #   ((1-P)/n_alternatives) * (
                #       r(alternative) + \gamma * V(alternative)
                #   ) for alternative in alternatives
                # )
                    values_all_actions.append(probability * (
                        destination_state.reward + \
                        discount * previous_values[destination_state] 
                    ) + sum([
                        (1.0 - probability) / 
                        (len(list(destination_states)) - 1) * (
                            alternative.reward + \
                            discount * previous_values[alternative]
                        ) for alternative in [
                            alternative for alternative in \
                            destination_states \
                            if alternative != destination_state
                        ]
                    ]))
                new_values[state] = max(values_all_actions)


                # calculate new delta
                delta = max(
                    [delta, abs(previous_values[state] - new_values[state])]
                )

            iteration += 1 
            previous_values = new_values

            if visualise:
                print(
                    f"Values for current iteration ({iteration}),",
                    f"with current delta of {delta}:"
                )
                print(self.values_in_maze_to_str(new_values))
        return previous_values

    @check_annotated
    def _determine_optimal_policy(
        self, 
        values: dict[State: float],
        discount: Annotated[float, FloatRange(0.0, 1.0)],
        probability: Annotated[float, FloatRange(0.0, 1.0)]=1.0
    )-> dict[State : Action]:
        """
        Determine optimal policy for given `self.maze`.

        This function performs the bellman function on the MDP
        in order to calculate the optimal policy 
        (using the value iteration)

        @param values: Dictionary with values for each state.
        @param discount: discount for future values/states
        @param probability: probability for any given action to succeed

        #return dict[State : Action] with optimal policy for each State.
        """
        actions = {state: None for state in self.maze.states.flatten()}

        for state in self.maze.states.flatten():
            # determine best action for state using $V(s) \leftarrow 
            # {argmax}_a \sum_{s',r}^{} 
            # p(s', r | s, a) [r + \gamma V(s')]$
            best_action_return = float("-inf")
            destionation_states = self.maze.get_destinations(state)
            for action, destination_state in destionation_states.items():
                # P * (r + \gamma * V(destination_state)) + sum(
                #   ((1-P)/n_alternatives) * (
                #       r(alternative) + \gamma * V(alternative)
                #   ) for alternative in alternatives
                # )
                new_action_return = probability * (
                    destination_state.reward + \
                    discount * values[destination_state]
                ) + sum([
                    (1.0 - probability) / 
                    (len(list(destionation_states.values())) - 1) * (
                        alternative.reward + discount * values[alternative]
                    ) for alternative in [
                        alternative for alternative in \
                        destionation_states.values() \
                        if alternative != destination_state
                    ]
                ])
                if  new_action_return> best_action_return:
                    best_action_return = new_action_return
                    actions[state] = action
        return actions

    def values_in_maze_to_str(self, values: dict[State : float])-> str:
        """
        Stringify values into maze matrix.

        Example:\n
        
        \n Values for current iteration (3), with current delta of 0.99:
        \n┌─────────────────────────┬─────────────────────────┐
        \n│ ( 0,2 ), v =  8.900000  │ ( 1,2 ), v = 37.214000  │
        \n├─────────────────────────┼─────────────────────────┤
        \n│ ( 0,1 ), v = 10.000000  │ ( 1,1 ), v =  8.900000  │
        \n├─────────────────────────┼─────────────────────────┤
        \n│ ( 0,0 ), v =  0.000000  │ ( 1,0 ), v = 10.000000  │
        \n└─────────────────────────┴─────────────────────────┘

        @return str with stringified values into maze matrix
        """
        # base case for the horizontal lines
        deviding_line = \
            f"{('─' * 25 + '┼') * (self.maze.states.shape[0] - 1)}"\
            f"{'─' * 25}"

        output = f"┌{deviding_line.replace('┼', '┬')}┐\n│ "

        # transform and reverse matrix, 
        # such that (0, 0) starts in the bottom left
        reversed_transformed_states = self.maze.states.T[::-1]
        for row in reversed_transformed_states[:-1]:
            for state in row.tolist():
                output += str(state).split('r')[0] + \
                    "v = {:^10.6f}\033[0m │ ".format(values[state])
            output += f"\n├{deviding_line}┤\n│ "

        # different formatting for last line
        for state in reversed_transformed_states[-1].tolist():
            output += str(state).split('r')[0] + \
                "v = {:^10.6f}\033[0m │ ".format(values[state])
        output += f"\n└{deviding_line.replace('┼', '┴')}┘"
        return output

    def select_action(self, state: State)-> Action:
        """
        Select action based on current policy.
        
        This policy works as follows:
        - select select the best action, given the MDP, and return it.

        @param state: Current State to perform action in.

        @return Action with Action to perform.
        """
        return self.actions[state]
    