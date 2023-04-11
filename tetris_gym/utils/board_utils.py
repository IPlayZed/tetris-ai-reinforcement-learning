import numpy as np


def binarize_board(board):
    return (np.array(board) > 0) * 1


def get_heights(board, binarize=False):
    if binarize:
        board = binarize_board(board)

    board_heights = board.shape[0]

    return (board_heights - board.argmax(axis=0)) % board_heights


def get_bumps(board):
    heights = get_heights(board)

    return get_bumps_from_heights(heights)


def get_bumps_from_heights(heights):
    bumps = []

    for i in range(len(heights) - 1):
        bumps.append(heights[i + 1] - heights[i])

    return bumps
