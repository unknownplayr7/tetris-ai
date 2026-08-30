WIDTH = 10
HEIGHT = 20


def new_board():
    return [[0 for _ in range(WIDTH)] for _ in range(HEIGHT)]


def valid(board, shape, x, y):
    for py, row in enumerate(shape):
        for px, cell in enumerate(row):
            if not cell:
                continue
            bx, by = x + px, y + py
            if bx < 0 or bx >= WIDTH or by >= HEIGHT:
                return False
            if by >= 0 and board[by][bx]:
                return False
    return True


def drop_y(board, shape, x):
    y = 0
    if not valid(board, shape, x, y):
        return None
    while valid(board, shape, x, y + 1):
        y += 1
    return y


def place(board, shape, x, y):
    result = [row[:] for row in board]
    for py, row in enumerate(shape):
        for px, cell in enumerate(row):
            if cell:
                result[y + py][x + px] = 1
    return result


def clear_lines(board):
    remaining = [row for row in board if not all(row)]
    cleared = HEIGHT - len(remaining)
    return [[0] * WIDTH for _ in range(cleared)] + remaining, cleared


def simulate_drop(board, shape, x):
    y = drop_y(board, shape, x)
    if y is None:
        return None, 0
    result = place(board, shape, x, y)
    return clear_lines(result)
