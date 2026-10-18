"""Heuristic Tetris placement search with optional one-piece lookahead."""
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
    """Higher scores represent safer, cleaner boards."""
    heights = column_heights(board)
    aggregate_height = sum(heights)
    bumpiness = sum(abs(a - b) for a, b in zip(heights, heights[1:]))
    holes = count_holes(board)
    max_height = max(heights, default=0)

    return (
        lines_cleared * 10.0
        - aggregate_height * 0.55
        - holes * 7.5
        - bumpiness * 0.35
        - max_height * 0.25
    )


def _best_follow_up_score(board, shape):
    best = float("-inf")
    for rotated in rotations(shape):
        shape_width = len(rotated[0])
        for x in range(WIDTH - shape_width + 1):
            result, lines = simulate_drop(board, rotated, x)
            if result is not None:
                best = max(best, evaluate(result, lines))
    return best


def find_best_move(board, shape, next_shape=None):
    """Return (rotation, shape, x, resulting_board, lines_cleared).

    If next_shape is provided, score each candidate by its immediate board
    quality plus a discounted estimate of the best placement for the next piece.
    """
    best_score = float("-inf")
    best_move = None

    for rotation_index, rotated in enumerate(rotations(shape)):
        shape_width = len(rotated[0])
        for x in range(WIDTH - shape_width + 1):
            result, lines = simulate_drop(board, rotated, x)
            if result is None:
                continue

            score = evaluate(result, lines)
            if next_shape is not None:
                follow_up = _best_follow_up_score(result, next_shape)
                if follow_up != float("-inf"):
                    score += 0.35 * follow_up

            if score > best_score:
                best_score = score
                best_move = (rotation_index, rotated, x, result, lines)

    return best_move
