import numpy as np
import pygame


WINDOW_SIZE = (800, 800)
FONT_SIZE = 20
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

def draw_matrix(
    data_matrix: np.ndarray, 
    colour_matrix: np.ndarray, 
    screen: pygame.display, 
    font: pygame.font
)-> None:
    """
    Draw any given matrix on given screen

    @param data_matrix: data to print in the squares
    @param colour_matrix: colours to give to the squares
    @param screen: display to display contents on
    @param font: font for text.
    """
    assert data_matrix.shape != colour_matrix.shape, \
        f"Wrong input dimensions! {data_matrix.shape} != {colour_matrix.shape}"
    
    screen.fill(WHITE)
    cell_width = WINDOW_SIZE[0] // data_matrix.shape[0]
    cell_height = WINDOW_SIZE[1] // data_matrix.shape[1]
    for x, row in enumerate(data_matrix):
        for y, data in enumerate(row):
            pygame.draw.rect(
                screen, 
                colour_matrix[:, x, y], 
                (y * cell_width, x * cell_height, cell_width, cell_height),
            )
            text_surface = font.render(str(data), True, WHITE)
            text_rect = text_surface.get_rect(
                center=(
                    y * cell_width + cell_width // 2, 
                    x * cell_height + cell_height // 2
                )
            )
            screen.blit(text_surface, text_rect)
    pygame.display.update()


def put_agent_colour_in_colour_matrix(
    colour_matrix: np.ndarray, 
    agent_coordinate: tuple[int, int],
    agent_colour: pygame.color
)-> np.ndarray:
    """
    Put agent colour into colour_matrix.

    @param colour_matrix: numpy array with colours
    @param agent_coordinate: x,y of agent
    @param agent_colour: colour of agent
    
    @return np.ndarray with altered colour matrix
    """
    try:
        colour_matrix[
            :, 
            colour_matrix.shape[1]-1-agent_coordinate[1], 
            agent_coordinate[0]
        ] = agent_colour
    except IndexError:
        raise IndexError(
            f"Index not found. Tried to alter index {agent_coordinate} "
            f"in matrix of shape {colour_matrix.shape}."
        )
    return colour_matrix
