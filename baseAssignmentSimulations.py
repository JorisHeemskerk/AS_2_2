import ASCII_table
import numpy as np
from tqdm import tqdm

from action import Action
from helper import Q_to_policy_np_matrix
from temporalDifferenceAgent import TemporalDifferenceAgent
from SARSAAgent import SARSAAgent 
from stochasticMaze import StochasticMaze
from QAgent import QAgent 
from doubleQAgent import DoubleQAgent 
from basePolicy import BasePolicy
from hardcodedOptimalPolicy import HardcodedOptimalPolicy
from stupidMaze import StupidMaze


def simulate_base_assignment_A(epochs: int)-> None:
    """
    Creates maze from assignment.
    Places TemporalDifferenceAgent in maze.
    gives agent optimal Policy.
    Perform Temporal Difference with gamma 1 and .5.
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
            print_result=False
        )
    agent.temporal_difference(
        alpha=.1,
        gamma=1,
        print_result=True
    )

    print(f"\033[32m{'─'*66}\n\t\tTemporal Difference, α=.1 γ=.5 epoch={epochs}\n{'─'*66}\033[0m")
    for _ in range(epochs-1):
        agent.temporal_difference(
            alpha=.1,
            gamma=.5,
            print_result=False
        )
    agent.temporal_difference(
            alpha=.1,
            gamma=.5,
            print_result=True
        )

def simulate_base_assignment_B(epochs: int)-> None:
    """
    Creates maze from assignment.
    Places SARSAAgent in maze.
    Perform SARSA with gamma 1 and .9.
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

    agent = SARSAAgent(maze, (2,0))

    print(f"\033[32m{'─'*43}\n\t\tMaze layout\n{'─'*43}\033[0m")
    print(maze.__str__(agent.current_coordinate))

    ################################
    #   SARSA with α=.1 ε=.1 γ=1   #
    ################################
    print(f"\033[32m{'─'*65}\n\t\tSARSA, α=.1 ε=.1 γ=1 epoch={epochs}\n{'─'*65}\033[0m")
    for _ in tqdm(range(epochs-1)):
        agent.sarsa(
            alpha=0.1,
            epsilon=0.1,
            gamma=1,
            print_result=False
        )
    agent.sarsa(
            alpha=0.1,
            epsilon=0.1,
            gamma=1,
            print_result=True
        )
    print(f"\033[32m{'─'*63}\n\t\tOptimal Policy derived from Q\n{'─'*63}\033[0m")
    policy_table = ASCII_table.ASCIITable(
        Q_to_policy_np_matrix(agent.Q).T[::-1],
        np.array([
            [
                ASCII_table.Colours.DEFAULT, 
                ASCII_table.Colours.DARK_YELLOW, 
                ASCII_table.Colours.DARK_YELLOW, 
                ASCII_table.Colours.RED    
            ], [
                ASCII_table.Colours.DEFAULT, 
                ASCII_table.Colours.DARK_YELLOW, 
                ASCII_table.Colours.BLUE, 
                ASCII_table.Colours.BLUE  
            ], [
                ASCII_table.Colours.DEFAULT, 
                ASCII_table.Colours.DARK_YELLOW, 
                ASCII_table.Colours.DARK_YELLOW, 
                ASCII_table.Colours.DEFAULT
            ], [
                ASCII_table.Colours.RED, 
                ASCII_table.Colours.DEFAULT, 
                ASCII_table.Colours.DARK_YELLOW, 
                ASCII_table.Colours.DEFAULT
            ],
        ])
    )
    policy_table.print()

    #################################
    #   SARSA with α=.1 ε=.1 γ=.9   #
    #################################
    print(f"\033[32m{'─'*65}\n\t\tSARSA, α=.1 ε=.1 γ=.9 epoch={epochs}\n{'─'*65}\033[0m")
    for _ in tqdm(range(epochs-1)):
        agent.sarsa(
            alpha=0.1,
            epsilon=0.1,
            gamma=0.9,
            print_result=False
        )
    agent.sarsa(
            alpha=0.1,
            epsilon=0.1,
            gamma=0.9,
            print_result=True
        )
    print(f"\033[32m{'─'*63}\n\t\tOptimal Policy derived from Q\n{'─'*63}\033[0m")
    policy_table = ASCII_table.ASCIITable(
        Q_to_policy_np_matrix(agent.Q).T[::-1],
        np.array([
            [
                ASCII_table.Colours.DEFAULT, 
                ASCII_table.Colours.DARK_YELLOW, 
                ASCII_table.Colours.DARK_YELLOW, 
                ASCII_table.Colours.RED    
            ], [
                ASCII_table.Colours.DEFAULT, 
                ASCII_table.Colours.DARK_YELLOW, 
                ASCII_table.Colours.BLUE, 
                ASCII_table.Colours.BLUE  
            ], [
                ASCII_table.Colours.DEFAULT, 
                ASCII_table.Colours.DARK_YELLOW, 
                ASCII_table.Colours.DARK_YELLOW, 
                ASCII_table.Colours.DEFAULT
            ], [
                ASCII_table.Colours.RED, 
                ASCII_table.Colours.DEFAULT, 
                ASCII_table.Colours.DARK_YELLOW, 
                ASCII_table.Colours.DEFAULT
            ],
        ])
    )
    policy_table.print()

def simulate_base_assignment_C(epochs: int)-> None:
    """
    Creates maze from assignment.
    Places QAgent in maze.
    Perform Q-learning with gamma 1 and .9.
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

    agent = QAgent(maze, (2,0))

    print(f"\033[32m{'─'*43}\n\t\tMaze layout\n{'─'*43}\033[0m")
    print(maze.__str__(agent.current_coordinate))

    ################################
    #   SARSA with α=.1 ε=.1 γ=1   #
    ################################
    print(f"\033[32m{'─'*70}\n\t\tQ-learning, α=.1 ε=.1 γ=1 epoch={epochs}\n{'─'*70}\033[0m")
    for _ in tqdm(range(epochs-1)):
        agent.Q_learning(
            alpha=0.1,
            epsilon=0.1,
            gamma=1,
            print_result=False
        )
    agent.Q_learning(
            alpha=0.1,
            epsilon=0.1,
            gamma=1,
            print_result=True
        )
    print(f"\033[32m{'─'*63}\n\t\tOptimal Policy derived from Q\n{'─'*63}\033[0m")
    policy_table = ASCII_table.ASCIITable(
        Q_to_policy_np_matrix(agent.Q).T[::-1],
        np.array([
            [
                ASCII_table.Colours.DEFAULT, 
                ASCII_table.Colours.DARK_YELLOW, 
                ASCII_table.Colours.DARK_YELLOW, 
                ASCII_table.Colours.RED    
            ], [
                ASCII_table.Colours.DEFAULT, 
                ASCII_table.Colours.DARK_YELLOW, 
                ASCII_table.Colours.BLUE, 
                ASCII_table.Colours.BLUE  
            ], [
                ASCII_table.Colours.DEFAULT, 
                ASCII_table.Colours.DARK_YELLOW, 
                ASCII_table.Colours.DARK_YELLOW, 
                ASCII_table.Colours.DEFAULT
            ], [
                ASCII_table.Colours.RED, 
                ASCII_table.Colours.DEFAULT, 
                ASCII_table.Colours.DARK_YELLOW, 
                ASCII_table.Colours.DEFAULT
            ],
        ])
    )
    policy_table.print()

    #################################
    #   SARSA with α=.1 ε=.1 γ=.9   #
    #################################
    print(f"\033[32m{'─'*70}\n\t\tQ-learning, α=.1 ε=.1 γ=.9 epoch={epochs}\n{'─'*70}\033[0m")
    for _ in tqdm(range(epochs-1)):
        agent.Q_learning(
            alpha=0.1,
            epsilon=0.1,
            gamma=0.9,
            print_result=False
        )
    agent.Q_learning(
            alpha=0.1,
            epsilon=0.1,
            gamma=0.9,
            print_result=True
        )
    print(f"\033[32m{'─'*63}\n\t\tOptimal Policy derived from Q\n{'─'*63}\033[0m")
    policy_table = ASCII_table.ASCIITable(
        Q_to_policy_np_matrix(agent.Q).T[::-1],
        np.array([
            [
                ASCII_table.Colours.DEFAULT, 
                ASCII_table.Colours.DARK_YELLOW, 
                ASCII_table.Colours.DARK_YELLOW, 
                ASCII_table.Colours.RED    
            ], [
                ASCII_table.Colours.DEFAULT, 
                ASCII_table.Colours.DARK_YELLOW, 
                ASCII_table.Colours.BLUE, 
                ASCII_table.Colours.BLUE  
            ], [
                ASCII_table.Colours.DEFAULT, 
                ASCII_table.Colours.DARK_YELLOW, 
                ASCII_table.Colours.DARK_YELLOW, 
                ASCII_table.Colours.DEFAULT
            ], [
                ASCII_table.Colours.RED, 
                ASCII_table.Colours.DEFAULT, 
                ASCII_table.Colours.DARK_YELLOW, 
                ASCII_table.Colours.DEFAULT
            ],
        ])
    )
    policy_table.print()

def simulate_base_assignment_EXTRA_D(epochs: int)-> None:
    """
    Creates stochastic maze.
    Places QAgent in maze.
    Perform Q-learning with gamma 1 and .9.
    """
    maze_shape = (4,4)
    rewards = np.array([
        [10,  -1,  -1,  -1],
        [-2,  -1,  -1,  -1],
        [-1,  -1, -10,  -1],
        [-1,  -1, -10,  40],
    ], dtype=int) # reward matrix from assignment
    maze = StochasticMaze(maze_shape, rewards, probability=0.1)
    maze.set_terminal((0,0))
    maze.set_terminal((3,3))   

    agent = QAgent(maze, (2,0))

    print(f"\033[32m{'─'*43}\n\t\tMaze layout\n{'─'*43}\033[0m")
    print(maze.__str__(agent.current_coordinate))

    ############################
    #   Q with α=.1 ε=.1 γ=1   #
    ############################
    print(f"\033[32m{'─'*70}\n\t\tQ-learning, α=.1 ε=.1 γ=1 epoch={epochs}\n{'─'*70}\033[0m")
    for _ in tqdm(range(epochs-1)):
        agent.Q_learning(
            alpha=0.1,
            epsilon=0.1,
            gamma=1,
            print_result=False
        )
    agent.Q_learning(
            alpha=0.1,
            epsilon=0.1,
            gamma=1,
            print_result=True
        )
    print(f"\033[32m{'─'*63}\n\t\tOptimal Policy derived from Q\n{'─'*63}\033[0m")
    policy_table = ASCII_table.ASCIITable(
        Q_to_policy_np_matrix(agent.Q).T[::-1],
        np.array([
            [
                ASCII_table.Colours.DEFAULT, 
                ASCII_table.Colours.DARK_YELLOW, 
                ASCII_table.Colours.DARK_YELLOW, 
                ASCII_table.Colours.RED    
            ], [
                ASCII_table.Colours.DEFAULT, 
                ASCII_table.Colours.DARK_YELLOW, 
                ASCII_table.Colours.BLUE, 
                ASCII_table.Colours.BLUE  
            ], [
                ASCII_table.Colours.DEFAULT, 
                ASCII_table.Colours.DARK_YELLOW, 
                ASCII_table.Colours.DARK_YELLOW, 
                ASCII_table.Colours.DEFAULT
            ], [
                ASCII_table.Colours.RED, 
                ASCII_table.Colours.DEFAULT, 
                ASCII_table.Colours.DARK_YELLOW, 
                ASCII_table.Colours.DEFAULT
            ],
        ])
    )
    policy_table.print()

    #############################
    #   Q with α=.1 ε=.1 γ=.9   #
    #############################
    print(f"\033[32m{'─'*70}\n\t\tQ-learning, α=.1 ε=.1 γ=.9 epoch={epochs}\n{'─'*70}\033[0m")
    for _ in tqdm(range(epochs-1)):
        agent.Q_learning(
            alpha=0.1,
            epsilon=0.1,
            gamma=0.9,
            print_result=False
        )
    agent.Q_learning(
            alpha=0.1,
            epsilon=0.1,
            gamma=0.9,
            print_result=True
        )
    print(f"\033[32m{'─'*63}\n\t\tOptimal Policy derived from Q\n{'─'*63}\033[0m")
    policy_table = ASCII_table.ASCIITable(
        Q_to_policy_np_matrix(agent.Q).T[::-1],
        np.array([
            [
                ASCII_table.Colours.DEFAULT, 
                ASCII_table.Colours.DARK_YELLOW, 
                ASCII_table.Colours.DARK_YELLOW, 
                ASCII_table.Colours.RED    
            ], [
                ASCII_table.Colours.DEFAULT, 
                ASCII_table.Colours.DARK_YELLOW, 
                ASCII_table.Colours.BLUE, 
                ASCII_table.Colours.BLUE  
            ], [
                ASCII_table.Colours.DEFAULT, 
                ASCII_table.Colours.DARK_YELLOW, 
                ASCII_table.Colours.DARK_YELLOW, 
                ASCII_table.Colours.DEFAULT
            ], [
                ASCII_table.Colours.RED, 
                ASCII_table.Colours.DEFAULT, 
                ASCII_table.Colours.DARK_YELLOW, 
                ASCII_table.Colours.DEFAULT
            ],
        ])
    )
    policy_table.print()

def simulate_base_assignment_EXTRA_E(epochs: int)-> None:
    """
    Creates StupidMaze.
    Places DoubleQAgent in maze.
    Perform double Q-learning with gamma 1 and .9.
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

    agent = DoubleQAgent(maze, (2,0))

    print(f"\033[32m{'─'*43}\n\t\tMaze layout\n{'─'*43}\033[0m")
    print(maze.__str__(agent.current_coordinate))

    ###################################
    #   Double Q with α=.1 ε=.1 γ=1   #
    ###################################
    print(f"\033[32m{'─'*77}\n\t\tDouble Q-learning, α=.1 ε=.1 γ=1 epoch={epochs}\n{'─'*77}\033[0m")
    for _ in tqdm(range(epochs-1)):
        agent.Q_learning(
            alpha=0.1,
            epsilon=0.1,
            gamma=1,
            print_result=False
        )
    agent.Q_learning(
            alpha=0.1,
            epsilon=0.1,
            gamma=1,
            print_result=True
        )
    print(f"\033[32m{'─'*63}\n\t\tOptimal Policy derived from Q\n{'─'*63}\033[0m")
    policy_table = ASCII_table.ASCIITable(
        Q_to_policy_np_matrix(agent.Q).T[::-1],
        np.array([
            [
                ASCII_table.Colours.DEFAULT, 
                ASCII_table.Colours.DARK_YELLOW, 
                ASCII_table.Colours.DARK_YELLOW, 
                ASCII_table.Colours.RED    
            ], [
                ASCII_table.Colours.DEFAULT, 
                ASCII_table.Colours.DARK_YELLOW, 
                ASCII_table.Colours.BLUE, 
                ASCII_table.Colours.BLUE  
            ], [
                ASCII_table.Colours.DEFAULT, 
                ASCII_table.Colours.DARK_YELLOW, 
                ASCII_table.Colours.DARK_YELLOW, 
                ASCII_table.Colours.DEFAULT
            ], [
                ASCII_table.Colours.RED, 
                ASCII_table.Colours.DEFAULT, 
                ASCII_table.Colours.DARK_YELLOW, 
                ASCII_table.Colours.DEFAULT
            ],
        ])
    )
    policy_table.print()

    ####################################
    #   Double Q with α=.1 ε=.1 γ=.9   #
    ####################################
    print(f"\033[32m{'─'*77}\n\t\tDouble Q-learning, α=.1 ε=.1 γ=.9 epoch={epochs}\n{'─'*77}\033[0m")
    for _ in tqdm(range(epochs-1)):
        agent.Q_learning(
            alpha=0.1,
            epsilon=0.1,
            gamma=0.9,
            print_result=False
        )
    agent.Q_learning(
            alpha=0.1,
            epsilon=0.1,
            gamma=0.9,
            print_result=True
        )
    print(f"\033[32m{'─'*63}\n\t\tOptimal Policy derived from Q\n{'─'*63}\033[0m")
    policy_table = ASCII_table.ASCIITable(
        Q_to_policy_np_matrix(agent.Q).T[::-1],
        np.array([
            [
                ASCII_table.Colours.DEFAULT, 
                ASCII_table.Colours.DARK_YELLOW, 
                ASCII_table.Colours.DARK_YELLOW, 
                ASCII_table.Colours.RED    
            ], [
                ASCII_table.Colours.DEFAULT, 
                ASCII_table.Colours.DARK_YELLOW, 
                ASCII_table.Colours.BLUE, 
                ASCII_table.Colours.BLUE  
            ], [
                ASCII_table.Colours.DEFAULT, 
                ASCII_table.Colours.DARK_YELLOW, 
                ASCII_table.Colours.DARK_YELLOW, 
                ASCII_table.Colours.DEFAULT
            ], [
                ASCII_table.Colours.RED, 
                ASCII_table.Colours.DEFAULT, 
                ASCII_table.Colours.DARK_YELLOW, 
                ASCII_table.Colours.DEFAULT
            ],
        ])
    )
    policy_table.print()
