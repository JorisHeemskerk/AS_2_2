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
import baseAssignmentSimulations as bas


def testing_setup(epochs: int)-> None:
    maze_shape = (5,7)
    probability = 0.1

    rewards = np.random.randint(-10, 10, size=maze_shape)

    maze = StochasticMaze(maze_shape, rewards, probability)
    maze.set_terminal((0,0))
    maze.set_terminal((2,6))   

    agent = QAgent(maze, (4,0))

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
        data=Q_to_policy_np_matrix(agent.Q).T[::-1]
    )
    policy_table.print()

def main()-> None:

    """ Base assignment, using CLI interface """
    # bas.simulate_base_assignment_A(epochs=500)
    # bas.simulate_base_assignment_B(epochs=1_000_000)
    # bas.simulate_base_assignment_C(epochs=1_000_000)
    # bas.simulate_base_assignment_EXTRA_D(epochs=1_000_000)
    # bas.simulate_base_assignment_EXTRA_E(epochs=1_000_000)

    testing_setup(epochs=100)

if __name__ == "__main__":
    main()
