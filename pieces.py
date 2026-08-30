import random

SHAPES = {
    "I": [[1, 1, 1, 1]],
    "O": [[1, 1], [1, 1]],
    "T": [[0, 1, 0], [1, 1, 1]],
    "S": [[0, 1, 1], [1, 1, 0]],
    "Z": [[1, 1, 0], [0, 1, 1]],
    "J": [[1, 0, 0], [1, 1, 1]],
    "L": [[0, 0, 1], [1, 1, 1]],
}


def rotate(shape):
    return [list(row) for row in zip(*shape[::-1])]


def rotations(shape):
    result = []
    current = shape
    for _ in range(4):
        if current not in result:
            result.append(current)
        current = rotate(current)
    return result


def random_piece():
    name = random.choice(list(SHAPES))
    return name, [row[:] for row in SHAPES[name]]
