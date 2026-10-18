# Board Inspector

A dependency-free command-line tool for inspecting a Tetris board. It prints the board and reports occupied cells, completed rows, each column's height, aggregate height, maximum height, holes, and bumpiness.

## Run

From the repository root:

```bash
python tools/board_inspector.py tools/sample_board.txt
```

Pass any text file containing exactly 20 rows of 10 cells. Use `#` for filled cells and `.` for empty cells. The parser also accepts `1`, `X`, and `@` as filled cells, and `0` or spaces as empty cells. Blank lines are ignored.

## Metrics

- **Aggregate height:** sum of all column heights.
- **Holes:** empty cells beneath the first occupied cell in each column.
- **Bumpiness:** sum of the absolute height differences between neighboring columns.

These are common board features used when evaluating Tetris positions. The script only analyzes the supplied snapshot; it does not modify the board or require third-party packages.
