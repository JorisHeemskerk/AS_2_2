import numpy as np

from action import Action
from temporalDifferenceAgent import TemporalDifferenceAgent
from basePolicy import BasePolicy
from hardcodedOptimalPolicy import HardcodedOptimalPolicy
from probabilityAgent import ProbabilityAgent
from stupidMaze import StupidMaze

def simulate_base_assignment_A(epochs: int=500)-> None:
    """
    Creates maze from assignment.
    Places TemporalDifferenceAgent in maze.
    gives agent optimal Policy.
    Perform Temporal Difference.
    """
    maze_shape = (4,4)
    rewards = np.array([
        [10,  -1,  -1,  -1],
        [-2,  -1,  -1,  -1],
        [-1,  -1, -10,  -1],
        [-1,  -1, -10,  40],
    ], dtype=int) # reward matrix from assignment
    maze = StupidMaze(maze_shape, rewards)
    maze.set_terminal((0,0))
    maze.set_terminal((3,3))   

    agent = TemporalDifferenceAgent(maze, BasePolicy(), (2,0))

    print(f"\033[32m{'─'*43}\n\t\tMaze layout\n{'─'*43}\033[0m")
    print(maze.__str__(agent.current_coordinate))

    print(f"\033[32m{'─'*49}\n\t\tOptimal Policy\n{'─'*49}\033[0m")
    optimal_actions = np.array([
        [Action.RIGHT, Action.RIGHT, Action.RIGHT, None       ],
        [Action.UP   , Action.UP   , Action.UP   , Action.UP  ],
        [Action.UP   , Action.UP   , Action.LEFT , Action.LEFT],
        [None        , Action.UP   , Action.UP   , Action.UP  ],
    ])[::-1].T
    policy = HardcodedOptimalPolicy(
        actions={
            state: action for state_row, action_row \
            in zip(maze.states, optimal_actions) for \
            state, action in zip(state_row, action_row)
        }
    )
    policy.visualise(maze)

    print(f"\033[32m{'─'*65}\n\t\tTemporal Difference, α=.1 γ=1 epoch={epochs}\n{'─'*65}\033[0m")
    agent.policy = policy
    for _ in range(epochs-1):
        agent.temporal_difference(
            alpha=.1,
            gamma=1,
            print_agent=False, 
            print_result=False
        )
    agent.temporal_difference(
        alpha=.1,
        gamma=1,
        print_agent=False, 
        print_result=True
    )

    print(f"\033[32m{'─'*66}\n\t\tTemporal Difference, α=.1 γ=.5 epoch={epochs}\n{'─'*66}\033[0m")
    for _ in range(epochs-1):
        agent.temporal_difference(
            alpha=.1,
            gamma=.5,
            print_agent=False, 
            print_result=False
        )
    agent.temporal_difference(
            alpha=.1,
            gamma=.5,
            print_agent=False, 
            print_result=True
        )

def simulate_base_assignment_EXTRA()-> None:
    """
    Creates maze from assignment.
    Places agent in maze.
    Print maze with agent.
    Perform value iteration, given a probability.
    Extract optimal policy.
    Print both.
    Have agent perform this optimal policy in maze, 
    using the probability.
    """
    maze_shape = (4,4)
    probability = 0.7

    rewards = np.array([
        [10,  -1,  -1,  -1],
        [-2,  -1,  -1,  -1],
        [-1,  -1, -10,  -1],
        [-1,  -1, -10,  40],
    ], dtype=int) # reward matrix from assignment

    maze = StupidMaze(maze_shape, rewards)
    maze.set_terminal((0,0))
    maze.set_terminal((3,3))

    agent = ProbabilityAgent(maze, BasePolicy(), (2,0), probability)

    print(f"\033[32m{'─'*43}\n\t\tMaze layout\n{'─'*43}\033[0m")
    print(maze.__str__(agent.current_coordinate))

    print(f"\033[32m{'─'*49}\n\t\tOptimizing Policy\n{'─'*49}\033[0m")
    policy = OptimalPolicy(
        maze=maze, 
        threshold=0.01,
        discount=1,
        probability=probability,
        visualise=True
    )

    print(f"\033[32m{'─'*45}\n\t\tAgent actions\n{'─'*45}\033[0m")
    # Assign policy to agent
    agent.policy = policy

    print(agent)
    # keep going until terminate state is reached
    while not maze[agent.current_coordinate].is_terminal:
        agent.act(True)
