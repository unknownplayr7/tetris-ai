#!/usr/bin/env python3
"""Inspect a Tetris board from a plain-text 20x10 grid.

Use '#' (or '1' or 'X') for occupied cells and '.' (or '0' or space) for
empty cells. The input file must contain exactly 20 rows of 10 cells.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

WIDTH = 10
HEIGHT = 20
FILLED = {"#", "1", "X", "x", "@"}
EMPTY = {".", "0", " "}


def parse_board(text: str) -> list[list[int]]:
    rows = [line.strip().replace("|", "") for line in text.splitlines() if line.strip()]
    if len(rows) != HEIGHT:
        raise ValueError(f"expected {HEIGHT} non-empty rows, found {len(rows)}")

    board = []
    for number, row in enumerate(rows, start=1):
        if len(row) != WIDTH:
            raise ValueError(f"row {number}: expected {WIDTH} cells, found {len(row)}")
        parsed = []
        for column, cell in enumerate(row, start=1):
            if cell in FILLED:
                parsed.append(1)
            elif cell in EMPTY:
                parsed.append(0)
            else:
                raise ValueError(
                    f"row {number}, column {column}: unsupported character {cell!r}"
                )
        board.append(parsed)
    return board


def inspect(board: list[list[int]]) -> dict[str, object]:
    heights = []
    holes = 0
    for x in range(WIDTH):
        occupied_rows = [y for y in range(HEIGHT) if board[y][x]]
        height = HEIGHT - occupied_rows[0] if occupied_rows else 0
        heights.append(height)
        if occupied_rows:
            holes += sum(
                1 for y in range(occupied_rows[0], HEIGHT) if not board[y][x]
            )

    return {
        "occupied": sum(map(sum, board)),
        "full_rows": sum(all(row) for row in board),
        "column_heights": heights,
        "aggregate_height": sum(heights),
        "holes": holes,
        "bumpiness": sum(abs(a - b) for a, b in zip(heights, heights[1:])),
        "max_height": max(heights, default=0),
    }


def render(board: list[list[int]]) -> str:
    border = "+" + "-" * WIDTH + "+"
    rows = ["".join("#" if cell else "." for cell in row) for row in board]
    return "\n".join([border, *("|" + row + "|" for row in rows), border])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print an ASCII Tetris board and useful board-quality metrics."
    )
    parser.add_argument("file", type=Path, help="text file containing a 20x10 board")
    args = parser.parse_args()

    try:
        board = parse_board(args.file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        print(f"board-inspector: {error}", file=sys.stderr)
        return 2

    metrics = inspect(board)
    print(render(board))
    print("\nBoard report")
    print("------------")
    print(f"Occupied cells:  {metrics['occupied']}")
    print(f"Full rows:       {metrics['full_rows']}")
    print(f"Column heights:  {' '.join(map(str, metrics['column_heights']))}")
    print(f"Aggregate height:{metrics['aggregate_height']:>4}")
    print(f"Maximum height:  {metrics['max_height']}")
    print(f"Holes:           {metrics['holes']}")
    print(f"Bumpiness:       {metrics['bumpiness']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
