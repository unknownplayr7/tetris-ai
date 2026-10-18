"""Polished Tetris game with a heuristic AI and a human-playable mode.

Run `python main.py` for AI autoplay, or `python main.py --mode human` to play.
Controls: arrows move, Up/X rotate, Z rotate counter-clockwise, Space hard-drops,
P pauses, Tab switches modes, and R restarts.
"""
import argparse
import random
import sys
import time

import pygame

from ai import find_best_move
from board import HEIGHT, WIDTH, clear_lines, drop_y, new_board, place, valid
from pieces import SHAPES, rotate

CELL = 30
BOARD_X = 38
BOARD_Y = 78
BOARD_W = WIDTH * CELL
BOARD_H = HEIGHT * CELL
PANEL_X = BOARD_X + BOARD_W + 34
WINDOW_W = PANEL_X + 252
WINDOW_H = BOARD_Y + BOARD_H + 36

BG = (10, 14, 27)
PANEL = (18, 25, 43)
PANEL_EDGE = (37, 51, 76)
GRID = (31, 42, 63)
TEXT = (235, 242, 255)
MUTED = (142, 158, 184)
ACCENT = (99, 220, 190)
DANGER = (255, 105, 130)
BLOCK_COLOR = (62, 180, 205)
PIECE_COLORS = {
    "I": (82, 220, 235),
    "O": (255, 207, 91),
    "T": (190, 125, 255),
    "S": (107, 225, 145),
    "Z": (255, 105, 130),
    "J": (100, 150, 255),
    "L": (255, 164, 92),
}


class PieceBag:
    """Seven-bag randomizer: each bag contains one of every tetromino."""

    def __init__(self):
        self.items = []

    def _refill(self):
        self.items = list(SHAPES)
        random.shuffle(self.items)

    def next_name(self):
        if not self.items:
            self._refill()
        return self.items.pop()

    def peek_name(self):
        if not self.items:
            self._refill()
        return self.items[-1]

    def peek_shape(self):
        name = self.peek_name()
        return name, [row[:] for row in SHAPES[name]]


class TetrisGame:
    def __init__(self, mode="ai"):
        self.mode = mode
        self.speed_factor = 1.0
        self.reset()

    def reset(self):
        self.board = new_board()
        self.score = 0
        self.lines = 0
        self.pieces_played = 0
        self.level = 1
        self.game_over = False
        self.paused = False
        self.bag = PieceBag()
        self.active_name = None
        self.shape = None
        self.x = 0
        self.y = 0
        self.target_y = 0
        self.ai_result = None
        self.ai_lines = 0
        self.spawn_piece()

    @property
    def gravity_ms(self):
        base = 55 if self.mode == "ai" else max(85, 520 - (self.level - 1) * 35)
        return max(28, int(base / self.speed_factor))

    def spawn_piece(self):
        self.active_name = self.bag.next_name()
        self.shape = [row[:] for row in SHAPES[self.active_name]]
        self.x = WIDTH // 2 - len(self.shape[0]) // 2
        self.y = 0
        self.pieces_played += 1
        self.ai_result = None

        if self.mode == "ai":
            _, next_shape = self.bag.peek_shape()
            move = find_best_move(self.board, self.shape, next_shape=next_shape)
            if move is None:
                self.game_over = True
                return
            _, self.shape, self.x, self.ai_result, self.ai_lines = move
            self.target_y = drop_y(self.board, self.shape, self.x)
            if self.target_y is None:
                self.game_over = True
        elif not valid(self.board, self.shape, self.x, self.y):
            self.game_over = True

    def ghost_y(self):
        y = self.y
        while valid(self.board, self.shape, self.x, y + 1):
            y += 1
        return y

    def _award_lines(self, cleared):
        self.lines += cleared
        self.level = 1 + self.lines // 10
        points = (0, 100, 300, 500, 800)
        self.score += points[min(cleared, 4)] * self.level

    def _finish_ai_piece(self):
        self.board = self.ai_result
        self._award_lines(self.ai_lines)
        self.spawn_piece()

    def lock_piece(self):
        self.board = place(self.board, self.shape, self.x, self.y)
        self.board, cleared = clear_lines(self.board)
        self._award_lines(cleared)
        self.spawn_piece()

    def tick(self):
        if self.game_over or self.paused:
            return
        if self.mode == "ai":
            if self.y < self.target_y:
                self.y += 1
            else:
                self._finish_ai_piece()
        elif valid(self.board, self.shape, self.x, self.y + 1):
            self.y += 1
        else:
            self.lock_piece()

    def move_horizontal(self, amount):
        new_x = self.x + amount
        if valid(self.board, self.shape, new_x, self.y):
            self.x = new_x

    def rotate_piece(self, turns=1):
        candidate = [row[:] for row in self.shape]
        for _ in range(turns):
            candidate = rotate(candidate)
        for kick in (0, -1, 1, -2, 2):
            if valid(self.board, candidate, self.x + kick, self.y):
                self.shape = candidate
                self.x += kick
                return

    def soft_drop(self):
        if valid(self.board, self.shape, self.x, self.y + 1):
            self.y += 1
            self.score += 1
        elif self.mode == "human":
            self.lock_piece()

    def hard_drop(self):
        if self.mode != "human" or self.game_over or self.paused:
            return
        self.y = self.ghost_y()
        self.score += max(0, self.y) * 2
        self.lock_piece()

    def switch_mode(self):
        self.mode = "human" if self.mode == "ai" else "ai"
        self.reset()


def text(surface, font, value, color, pos):
    surface.blit(font.render(str(value), True, color), pos)


def draw_block(surface, x, y, color, size=CELL, ghost=False):
    rect = pygame.Rect(x, y, size, size).inflate(-4, -4)
    if ghost:
        pygame.draw.rect(surface, color, rect, width=2, border_radius=5)
        return
    pygame.draw.rect(surface, color, rect, border_radius=5)
    shine = tuple(min(255, channel + 35) for channel in color)
    pygame.draw.line(surface, shine, (rect.x + 5, rect.y + 4),
                     (rect.right - 5, rect.y + 4), 2)
    pygame.draw.line(surface, tuple(int(c * 0.65) for c in color),
                     (rect.x + 5, rect.bottom - 4),
                     (rect.right - 5, rect.bottom - 4), 2)


def draw_piece(surface, shape, x, y, color, ghost=False, cell=CELL):
    for py, row in enumerate(shape):
        for px, filled in enumerate(row):
            if filled:
                draw_block(
                    surface, BOARD_X + (x + px) * cell,
                    BOARD_Y + (y + py) * cell, color, cell, ghost,
                )


def draw_game(screen, game, fonts):
    title_font, heading_font, body_font, small_font = fonts
    screen.fill(BG)

    # Header
    text(screen, title_font, "TETRIS", TEXT, (BOARD_X, 22))
    text(screen, small_font, "AI LAB  /  PLACEMENT ENGINE", MUTED, (BOARD_X + 122, 31))
    mode_label = "AI AUTOPLAY" if game.mode == "ai" else "HUMAN MODE"
    mode_color = ACCENT if game.mode == "ai" else (255, 207, 91)
    pill = pygame.Rect(PANEL_X, 24, 154, 31)
    pygame.draw.rect(screen, PANEL, pill, border_radius=12)
    pygame.draw.rect(screen, PANEL_EDGE, pill, width=1, border_radius=12)
    text(screen, small_font, mode_label, mode_color, (pill.x + 14, pill.y + 7))

    # Board frame and grid
    frame = pygame.Rect(BOARD_X - 8, BOARD_Y - 8, BOARD_W + 16, BOARD_H + 16)
    pygame.draw.rect(screen, PANEL, frame, border_radius=12)
    pygame.draw.rect(screen, PANEL_EDGE, frame, width=2, border_radius=12)
    pygame.draw.rect(screen, (7, 10, 20), (BOARD_X, BOARD_Y, BOARD_W, BOARD_H))
    for y in range(HEIGHT):
        for x in range(WIDTH):
            rect = pygame.Rect(BOARD_X + x * CELL, BOARD_Y + y * CELL, CELL, CELL)
            pygame.draw.rect(screen, GRID, rect, width=1)
            if game.board[y][x]:
                draw_block(screen, rect.x, rect.y, BLOCK_COLOR)

    if game.shape is not None and not game.game_over:
        color = PIECE_COLORS[game.active_name]
        if game.mode == "human":
            draw_piece(screen, game.shape, game.x, game.ghost_y(), color, ghost=True)
        draw_piece(screen, game.shape, game.x, game.y, color)

    # Side panel cards
    card = pygame.Rect(PANEL_X, 78, 236, 166)
    pygame.draw.rect(screen, PANEL, card, border_radius=12)
    pygame.draw.rect(screen, PANEL_EDGE, card, width=1, border_radius=12)
    text(screen, small_font, "NEXT PIECE", MUTED, (card.x + 16, card.y + 14))
    next_name, next_shape = game.bag.peek_shape()
    preview_cell = 22
    shape_w = len(next_shape[0]) * preview_cell
    shape_h = len(next_shape) * preview_cell
    start_x = card.x + (card.width - shape_w) // 2
    start_y = card.y + 48 + (74 - shape_h) // 2
    for py, row in enumerate(next_shape):
        for px, filled in enumerate(row):
            if filled:
                draw_block(screen, start_x + px * preview_cell,
                           start_y + py * preview_cell,
                           PIECE_COLORS[next_name], preview_cell)

    stats = pygame.Rect(PANEL_X, 258, 236, 194)
    pygame.draw.rect(screen, PANEL, stats, border_radius=12)
    pygame.draw.rect(screen, PANEL_EDGE, stats, width=1, border_radius=12)
    text(screen, small_font, "RUN STATS", MUTED, (stats.x + 16, stats.y + 14))
    labels = [
        ("SCORE", f"{game.score:,}"),
        ("LINES", game.lines),
        ("LEVEL", game.level),
        ("PIECES", game.pieces_played),
    ]
    for index, (label, value) in enumerate(labels):
        y = stats.y + 45 + index * 34
        text(screen, small_font, label, MUTED, (stats.x + 16, y))
        text(screen, body_font, value, TEXT, (stats.x + 132, y - 4))

    controls = pygame.Rect(PANEL_X, 466, 236, 238)
    pygame.draw.rect(screen, PANEL, controls, border_radius=12)
    pygame.draw.rect(screen, PANEL_EDGE, controls, width=1, border_radius=12)
    text(screen, small_font, "CONTROLS", MUTED, (controls.x + 16, controls.y + 14))
    if game.mode == "ai":
        hints = [
            ("TAB", "Switch to human"),
            ("P / SPACE", "Pause / resume"),
            ("R", "Restart run"),
            ("+ / -", "Adjust speed"),
            ("AI", "2-piece lookahead"),
        ]
    else:
        hints = [
            ("← / →", "Move"),
            ("↑ / X", "Rotate"),
            ("Z", "Rotate left"),
            ("SPACE", "Hard drop"),
            ("↓", "Soft drop"),
            ("TAB", "Switch to AI"),
        ]
    for index, (key, description) in enumerate(hints):
        y = controls.y + 43 + index * 29
        text(screen, small_font, key, ACCENT, (controls.x + 16, y))
        text(screen, small_font, description, TEXT, (controls.x + 91, y))

    if game.paused or game.game_over:
        overlay = pygame.Surface((BOARD_W, BOARD_H), pygame.SRCALPHA)
        overlay.fill((5, 8, 18, 190))
        screen.blit(overlay, (BOARD_X, BOARD_Y))
        message = "GAME OVER" if game.game_over else "PAUSED"
        sub = "Press R to restart" if game.game_over else "Press P to continue"
        big = title_font.render(message, True, DANGER if game.game_over else TEXT)
        small = body_font.render(sub, True, TEXT)
        screen.blit(big, big.get_rect(center=(BOARD_X + BOARD_W // 2, BOARD_Y + BOARD_H // 2 - 12)))
        screen.blit(small, small.get_rect(center=(BOARD_X + BOARD_W // 2, BOARD_Y + BOARD_H // 2 + 24)))

    pygame.display.flip()


def main():
    parser = argparse.ArgumentParser(description="Play Tetris or watch the heuristic AI.")
    parser.add_argument("--mode", choices=("ai", "human"), default="ai")
    args = parser.parse_args()

    pygame.init()
    pygame.display.set_caption("Tetris AI Lab")
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    clock = pygame.time.Clock()
    fonts = (
        pygame.font.SysFont("menlo,consolas,monospace", 30, bold=True),
        pygame.font.SysFont("menlo,consolas,monospace", 23, bold=True),
        pygame.font.SysFont("menlo,consolas,monospace", 19),
        pygame.font.SysFont("menlo,consolas,monospace", 15),
    )

    game = TetrisGame(args.mode)
    last_tick = pygame.time.get_ticks()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    game.reset()
                    last_tick = pygame.time.get_ticks()
                elif event.key == pygame.K_TAB:
                    game.switch_mode()
                    last_tick = pygame.time.get_ticks()
                elif event.key == pygame.K_p or (event.key == pygame.K_SPACE and game.mode == "ai"):
                    game.paused = not game.paused
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    game.speed_factor = min(4.0, game.speed_factor * 1.25)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    game.speed_factor = max(0.25, game.speed_factor / 1.25)
                elif game.mode == "human" and not game.paused and not game.game_over:
                    if event.key == pygame.K_LEFT:
                        game.move_horizontal(-1)
                    elif event.key == pygame.K_RIGHT:
                        game.move_horizontal(1)
                    elif event.key == pygame.K_DOWN:
                        game.soft_drop()
                    elif event.key in (pygame.K_UP, pygame.K_x):
                        game.rotate_piece()
                    elif event.key == pygame.K_z:
                        game.rotate_piece(3)
                    elif event.key == pygame.K_SPACE:
                        game.hard_drop()

        now = pygame.time.get_ticks()
        if now - last_tick >= game.gravity_ms:
            game.tick()
            last_tick = now

        draw_game(screen, game, fonts)
        clock.tick(60)

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
