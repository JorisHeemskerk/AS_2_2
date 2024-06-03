import numpy as np

from state import State


def state_dict_to_np_matrix(
    state_dict: dict[State : any], 
    empty_cell_contents: str='',
    prefix: str=''
)-> np.ndarray:
    """
    Helper function to convert state dictionaries to matrices

    @param state_dict: dict with state and value that can be stringified
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
