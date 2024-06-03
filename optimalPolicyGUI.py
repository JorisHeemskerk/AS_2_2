import numpy as np
import pygame
from typing import Annotated

from action import Action
from optimalPolicy import OptimalPolicy
from floatRange import FloatRange, check_annotated
from baseMaze import BaseMaze
from state import State
from utils import draw_matrix, put_agent_colour_in_colour_matrix, BLACK, WINDOW_SIZE
        

class OptimalPolicyGUI(OptimalPolicy):
    """
    OptimalPolicyGUI
    
    Exactly like the OptimalPolicy class, but it has a GUI for the 
    value iteration and the optimal policy extraction.
    """

    @check_annotated
    def __init__(
        self, 
        maze: BaseMaze, 
        threshold: Annotated[float, FloatRange(0.0, float("inf"))],
        discount: Annotated[float, FloatRange(0.0, 1.0)],
        colour_matrix: np.ndarray,
        probability: Annotated[float, FloatRange(0.0, 1.0)]=1.0
    )-> None:
        """
        @var $maze
        **Maze** with MDP information

        @var $actions
        **dict[state : action]** 
        dictionary with states mapping to optimal actions.  

        @var $font
        **pygame.font**
        pygame font for GUI
        
        @var $screen
        **pygame.display**
        pygame display for GUI
        
        """
        pygame.init()
        self.font = pygame.font.SysFont(None, 20)
        self.screen = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption(
            "Calculating Optimal Policy, Iteration = 0, Delta = 0"
        )
        self.colour_matrix = colour_matrix
        self.maze = maze
        self.actions = self._determine_optimal_policy(
            self._value_iteration(
                threshold, 
                discount,
                probability, 
                False
            ), 
            discount,
            probability
        )

    @check_annotated
    def _value_iteration(
        self, 
        threshold: Annotated[float, FloatRange(0.0, float("inf"))],
        discount: Annotated[float, FloatRange(0.0, 1.0)],
        probability: Annotated[float, FloatRange(0.0, 1.0)]=1.0,
        visualise: bool=False
    )-> dict[State : float]:
        """
        Value iteration with GUI

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
        

        data_matrix = np.array([
            [f"r = {state.reward} | v = 0" for state in row]
            for row in self.maze.states
        ]).T[::-1]
        draw_matrix(data_matrix, self.colour_matrix, self.screen, self.font)

        running = True
        while running:
            pygame.time.delay(1000)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            if delta >= threshold:
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
                    delta = max([
                        delta, abs(previous_values[state] - new_values[state])
                    ])

                iteration += 1 
                previous_values = new_values

                pygame.display.set_caption(
                    f"Calculating Optimal Policy,"
                    f" Iteration = {iteration}, Delta = {delta}"
                )
                data_matrix = np.array([
                    [
                        f"r = {state.reward} | v = {previous_values[state]}" \
                        for state in row
                    ]
                    for row in self.maze.states
                ]).T[::-1]
                draw_matrix(data_matrix, self.colour_matrix, self.screen, self.font)

        pygame.quit()
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

        pygame.init()
        self.font = pygame.font.SysFont(None, 20)
        self.screen = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption(
            "Displaying optimal policy"
        )
        action_to_arrow = {
            Action.UP : "^",
            Action.DOWN : "v",
            Action.LEFT : "<",
            Action.RIGHT : ">",
            None: "X"
        }
        data_matrix = np.array([
            [
                f"r = {state.reward} | v = {round(values[state], 2)} |"
                f" a = {action_to_arrow[actions[state]]}"
                for state in row
            ]
            for row in self.maze.states
        ]).T[::-1]
        draw_matrix(data_matrix, self.colour_matrix, self.screen, self.font)

        running = True
        while running:
            pygame.time.delay(100)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
        pygame.quit()
        return actions
