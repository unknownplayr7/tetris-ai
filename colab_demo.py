"""Colab-friendly Tetris AI demo.

Runs the existing heuristic AI and renders the board directly in a
Google Colab notebook using Unicode blocks. No desktop window is needed.
"""

import time

from IPython.display import clear_output

from board import WIDTH, new_board
from pieces import random_piece
from ai import find_best_move


BLOCK = "██"
EMPTY = "  "


def print_board(board, pieces_played, lines_total, x, rotation_index, piece_name):
    clear_output(wait=True)

    print("TETRIS AI")
    print()
    print("+" + "--" * WIDTH + "+")

    for row in board:
        print("|" + "".join(BLOCK if cell else EMPTY for cell in row) + "|")

    print("+" + "--" * WIDTH + "+")
    print(f"Pieces played: {pieces_played}")
    print(f"Lines cleared: {lines_total}")
    print(f"Piece: {piece_name}")
    print(f"Move: x={x}, rotation={rotation_index}")


def main(delay=0.15):
    board = new_board()
    lines_total = 0
    pieces_played = 0

    while True:
        piece_name, shape = random_piece()
        move = find_best_move(board, shape)

        if move is None:
            clear_output(wait=True)
            print("TETRIS AI")
            print()
            print("GAME OVER")
            print(f"Pieces played: {pieces_played}")
            print(f"Lines cleared: {lines_total}")
            break

        rotation_index, rotated, x, result, lines = move
        board = result
        lines_total += lines
        pieces_played += 1

        print_board(
            board,
            pieces_played,
            lines_total,
            x,
            rotation_index,
            piece_name,
        )
        time.sleep(delay)


if __name__ == "__main__":
    main()
