from board import WIDTH, simulate_drop
from pieces import rotations


def column_heights(board):
    heights = []
    for x in range(WIDTH):
        height = 0
        for y in range(len(board)):
            if board[y][x]:
                height = len(board) - y
                break
        heights.append(height)
    return heights


def count_holes(board):
    holes = 0
    for x in range(WIDTH):
        seen_block = False
        for y in range(len(board)):
            if board[y][x]:
                seen_block = True
            elif seen_block:
                holes += 1
    return holes


def evaluate(board, lines_cleared):
    heights = column_heights(board)
    aggregate_height = sum(heights)
    bumpiness = sum(abs(heights[i] - heights[i + 1]) for i in range(WIDTH - 1))
    holes = count_holes(board)

    return (
        lines_cleared * 10.0
        - aggregate_height * 0.55
        - holes * 7.5
        - bumpiness * 0.35
        - max(heights) * 0.25
    )


def find_best_move(board, shape):
    best_score = float("-inf")
    best_move = None

    for rotation_index, rotated in enumerate(rotations(shape)):
        shape_width = len(rotated[0])
        for x in range(WIDTH - shape_width + 1):
            result, lines = simulate_drop(board, rotated, x)
            if result is None:
                continue

            score = evaluate(result, lines)
            if score > best_score:
                best_score = score
                best_move = (rotation_index, rotated, x, result, lines)

    return best_move
