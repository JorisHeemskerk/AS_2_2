import numpy as np

from state import State
from action import Action


def state_dict_to_np_matrix(
    state_dict: dict[State : any], 
    empty_cell_contents: str='',
    prefix: str=''
)-> np.ndarray:
    """
    Helper function to convert state dictionaries to matrices

    @param state_dict: dict with state and value that can be stringified
    @param empty_cell_contents: what to put in empty cells
    @param prefix: prefix to put in front of string.
    """
    sorted_states = sorted(state_dict.keys())
    
    max_x = max(state.position[0] for state in sorted_states)
    max_y = max(state.position[1] for state in sorted_states)

    matrix = np.full((max_x + 1, max_y + 1), None, dtype=object)
    matrix.fill((empty_cell_contents,))
    for state in sorted_states:
        matrix[state.position[0], state.position[1]] = \
            (f"{prefix}{str(state_dict[state])}",)

    return matrix

def Q_to_np_matrix(
    Q: dict[State : dict[Action : float]], 
    rounding_digits: int=2,
    empty_cell_contents: str='',
)-> np.ndarray:
    """
    Helper function to convert Q dictionaries to matrices

    @param Q: dict with state to dict with action to floats
    @param rounding_digits: number of digits to round values to
    @param empty_cell_contents: what to put in empty cells
    """
    sorted_states = sorted(Q.keys())

    max_x = max(state.position[0] for state in sorted_states)
    max_y = max(state.position[1] for state in sorted_states)

    matrix = np.full((max_x + 1, max_y + 1), None, dtype=object)
    matrix.fill((empty_cell_contents,))

    # Populate matrix with tuples
    for state in sorted_states:
        matrix[state.position[0], state.position[1]] = (
            f"{str(round(Q[state][Action.UP], rounding_digits))}",
            f"{str(round(Q[state][Action.LEFT], rounding_digits))}" \
                f"   {str(round(Q[state][Action.RIGHT], rounding_digits))}",
            f"{str(round(Q[state][Action.DOWN], rounding_digits))}",
        )
        
    return matrix

def Q_to_policy_np_matrix(
    Q: dict[State : dict[Action : float]], 
    empty_cell_contents: str='',
)-> np.ndarray:
    """
    Helper function to convert Q dictionaries, to policies, to matrices

    @param Q: dict with state to dict with action to floats
    @param empty_cell_contents: what to put in empty cells
    """
    sorted_states = sorted(Q.keys())

    max_x = max(state.position[0] for state in sorted_states)
    max_y = max(state.position[1] for state in sorted_states)

    matrix = np.full((max_x + 1, max_y + 1), None, dtype=object)
    matrix.fill((empty_cell_contents,))

    action_to_arrow = {
        Action.UP : "▲",
        Action.DOWN : "▼",
        Action.LEFT : "◄",
        Action.RIGHT : "►",
        None: "✕"
    }
    # Populate matrix with tuples
    for state in sorted_states:
        matrix[state.position[0], state.position[1]] = (
            action_to_arrow[max(Q[state], key=Q[state].get)],
        ) if not state.is_terminal else (action_to_arrow[None], )
        
    return matrix