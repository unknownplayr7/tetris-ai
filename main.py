import pygame
from board import WIDTH, HEIGHT, new_board
from pieces import random_piece
from ai import find_best_move

CELL = 28
PADDING = 40
PANEL = 180
WINDOW_W = WIDTH * CELL + PADDING * 2 + PANEL
WINDOW_H = HEIGHT * CELL + PADDING * 2


def draw(screen, board, score, lines, game_over):
    screen.fill((20, 20, 28))
    ox, oy = PADDING, PADDING

    for y in range(HEIGHT):
        for x in range(WIDTH):
            rect = pygame.Rect(ox + x * CELL, oy + y * CELL, CELL, CELL)
            pygame.draw.rect(screen, (42, 42, 54), rect, 1)
            if board[y][x]:
                pygame.draw.rect(screen, (80, 190, 255), rect.inflate(-3, -3))

    font = pygame.font.SysFont(None, 30)
    small = pygame.font.SysFont(None, 24)
    tx = ox + WIDTH * CELL + 30
    screen.blit(font.render("TETRIS AI", True, (240, 240, 245)), (tx, oy))
    screen.blit(small.render(f"Score: {score}", True, (220, 220, 225)), (tx, oy + 55))
    screen.blit(small.render(f"Lines: {lines}", True, (220, 220, 225)), (tx, oy + 85))
    screen.blit(small.render("Heuristic AI", True, (160, 200, 255)), (tx, oy + 130))

    if game_over:
        text = font.render("GAME OVER", True, (255, 100, 100))
        screen.blit(text, (tx, oy + 180))

    pygame.display.flip()


def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Tetris AI")
    clock = pygame.time.Clock()

    board = new_board()
    score = 0
    total_lines = 0
    game_over = False
    move_delay = 180
    last_move = pygame.time.get_ticks()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    board = new_board()
                    score = 0
                    total_lines = 0
                    game_over = False

        now = pygame.time.get_ticks()
        if not game_over and now - last_move >= move_delay:
            _, shape = random_piece()
            move = find_best_move(board, shape)

            if move is None:
                game_over = True
            else:
                _, _, _, board, cleared = move
                total_lines += cleared
                score += 100 + cleared * cleared * 100

            last_move = now

        draw(screen, board, score, total_lines, game_over)
        clock.tick(60)


if __name__ == "__main__":
    main()
