import sys

import pygame

from utils import *
from grid import Grid
from searching_algorithms import *

pygame.init()

WIDTH = 1200
HEIGHT = 800
GRID_SIZE = 700
SIDEBAR_WIDTH = WIDTH - GRID_SIZE

COLORS = {
    'BACKGROUND': (245, 240, 245),
    'SIDEBAR': (235, 225, 235),
    'BUTTON': (210, 190, 220),
    'BUTTON_HOVER': (200, 180, 210),
    'BUTTON_ACTIVE': (160, 130, 190),
    'TEXT': (70, 60, 80),
    'ACCENT': (180, 150, 220),
    'SUCCESS': (170, 225, 200),
    'DANGER': (255, 170, 170),
    'RED': (255, 160, 160),
    'GREEN': (180, 245, 210),
    'BLUE': (170, 210, 255),
    'YELLOW': (255, 245, 180),
    'WHITE': (255, 250, 255),
    'BLACK': (80, 70, 90),
    'PURPLE': (200, 170, 240),
    'ORANGE': (255, 200, 160),
    'GREY': (210, 200, 220),
    'TURQUOISE': (160, 230, 230),
    'PATH': (190, 160, 230)
}

class Button:
    def __init__(self, x, y, width, height, text, color, hover_color, text_color):
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color
        self.hover_color = hover_color
        self.text_color = text_color
        self.text = text
        self.is_hovered = False
        self.is_active = False

    def draw(self, win, font):
        color = self.hover_color if self.is_hovered else self.color
        if self.is_active:
            color = COLORS['BUTTON_ACTIVE']

        pygame.draw.rect(win, color, self.rect, border_radius = 8)

        if self.is_active:
            pygame.draw.rect(win, COLORS['ACCENT'], self.rect, 2, border_radius = 8)

        text_surface = font.render(self.text, True, self.text_color)
        text_rect = text_surface.get_rect(center = self.rect.center)
        win.blit(text_surface, text_rect)

    def check_hover(self, pos):
        self.is_hovered = self.rect.collidepoint(pos)
        return self.is_hovered

    def is_clicked(self, pos):
        return self.rect.collidepoint(pos)

def draw_ui(win, buttons, selected_algorithm, stats, font, title_font):
    sidebar_rect = pygame.Rect(GRID_SIZE, 0, SIDEBAR_WIDTH, HEIGHT)
    pygame.draw.rect(win, COLORS['SIDEBAR'], sidebar_rect)

    title = title_font.render("Path Finder", True, COLORS['ACCENT'])
    win.blit(title, (GRID_SIZE + 20, 20))

    subtitle = font.render("AI Search Algorithms", True, COLORS['TEXT'])
    win.blit(subtitle, (GRID_SIZE + 20, 65))

    pygame.draw.line(win, COLORS['GREY'], (GRID_SIZE + 20, 100), (WIDTH - 20, 100), 2)

    algo_label = font.render("Select Algorithm:", True, COLORS['TEXT'])
    win.blit(algo_label, (GRID_SIZE + 20, 120))

    for button in buttons:
        button.draw(win, font)

    stats_y = 550
    pygame.draw.line(win, COLORS['GREY'], (GRID_SIZE + 20, stats_y), (WIDTH - 20, stats_y), 2)

    stats_title = font.render("Statistics:", True, COLORS['TEXT'])
    win.blit(stats_title, (GRID_SIZE + 20, stats_y + 15))

    if stats:
        status_color = COLORS['SUCCESS'] if stats.get('found', False) else COLORS['DANGER']
        status_text = "Path Found!" if stats.get('found', False) else "No Path"
        status = font.render(f"Status: {status_text}", True, status_color)
        win.blit(status, (GRID_SIZE + 20, stats_y + 50))

    controls_y = 650
    pygame.draw.line(win, COLORS['GREY'], (GRID_SIZE + 20, controls_y), (WIDTH - 20, controls_y), 2)

    controls_title = font.render("Controls:", True, COLORS['TEXT'])
    win.blit(controls_title, (GRID_SIZE + 20, controls_y + 15))

    controls = [
        "Left Click: Draw/Place",
        "Right Click: Erase",
        "Space: Start Search",
        "C: Clear Board",
        "R: Generate Maze"
    ]

    for i, control in enumerate(controls):
        text = font.render(control, True, COLORS['TEXT'])
        win.blit(text, (GRID_SIZE + 20, controls_y + 50 + i * 25))

def generate_random_maze(grid):
    import random
    for row in grid.grid:
        for spot in row:
            if random.random() < 0.3:  # 30% chance of barrier
                if not spot.is_start() and not spot.is_end():
                    spot.make_barrier()


def main():
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Visual Search - AI Path Finding")

    font = pygame.font.Font(None, 24)
    title_font = pygame.font.Font(None, 36)

    ROWS = 50
    grid_surface = pygame.Surface((GRID_SIZE, HEIGHT))
    grid = Grid(grid_surface, ROWS, ROWS, GRID_SIZE, HEIGHT)

    algorithms = [
        ("BFS", bfs),
        ("DFS", dfs),
        ("DLS", lambda d, g, s, e: dls(d, g, s, e, 20)),
        ("A*", astar),
        ("UCS", ucs),
        ("Greedy", greedy),
        ("IDS", lambda d, g, s, e: ids(d, g, s, e, 30)),
        ("IDA*", ida_star)
    ]

    buttons = []
    button_y = 150
    for i, (name, func) in enumerate(algorithms):
        button = Button(
            GRID_SIZE + 20,
            button_y + i * 45,
            SIDEBAR_WIDTH - 40,
            35,
            name,
            COLORS['BUTTON'],
            COLORS['BUTTON_HOVER'],
            COLORS['TEXT']
        )
        buttons.append((button, func))

    clear_button = Button(GRID_SIZE + 20, 500, (SIDEBAR_WIDTH - 50) // 2, 35, "Clear", COLORS['DANGER'],
                          (255, 100, 120), COLORS['TEXT'])
    maze_button = Button(GRID_SIZE + 30 + (SIDEBAR_WIDTH - 50) // 2, 500, (SIDEBAR_WIDTH - 50) // 2, 35, "Maze",
                         COLORS['SUCCESS'], (100, 220, 140), COLORS['TEXT'])

    selected_algorithm = 0
    buttons[selected_algorithm][0].is_active = True

    start = None
    end = None
    stats = {}

    run = True
    clock = pygame.time.Clock()

    sidebar_needs_redraw = True

    while run:
        clock.tick(60)
        mouse_pos = pygame.mouse.get_pos()

        hover_changed = False
        for button, _ in buttons:
            old_hover = button.is_hovered
            button.check_hover(mouse_pos)
            if old_hover != button.is_hovered:
                hover_changed = True

        old_clear_hover = clear_button.is_hovered
        old_maze_hover = maze_button.is_hovered
        clear_button.check_hover(mouse_pos)
        maze_button.check_hover(mouse_pos)
        if old_clear_hover != clear_button.is_hovered or old_maze_hover != maze_button.is_hovered:
            hover_changed = True

        if hover_changed:
            sidebar_needs_redraw = True

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if pygame.mouse.get_pressed()[0]:
                if mouse_pos[0] < GRID_SIZE:
                    col, row = grid.get_clicked_pos(mouse_pos)
                    if 0 <= row < ROWS and 0 <= col < ROWS:
                        spot = grid.grid[row][col]
                        if not start and spot != end:
                            start = spot
                            start.make_start()
                        elif not end and spot != start:
                            end = spot
                            end.make_end()
                        elif spot != end and spot != start:
                            spot.make_barrier()
                else:
                    for i, (button, _) in enumerate(buttons):
                        if button.is_clicked(mouse_pos):
                            buttons[selected_algorithm][0].is_active = False
                            selected_algorithm = i
                            buttons[selected_algorithm][0].is_active = True
                            sidebar_needs_redraw = True

                    if clear_button.is_clicked(mouse_pos):
                        start = None
                        end = None
                        grid.reset()
                        stats = {}
                        sidebar_needs_redraw = True

                    if maze_button.is_clicked(mouse_pos):
                        generate_random_maze(grid)

            elif pygame.mouse.get_pressed()[2]:
                if mouse_pos[0] < GRID_SIZE:
                    col, row = grid.get_clicked_pos(mouse_pos)
                    if 0 <= row < ROWS and 0 <= col < ROWS:
                        spot = grid.grid[row][col]
                        spot.reset()
                        if spot == start:
                            start = None
                        elif spot == end:
                            end = None

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and start and end:
                    # Update neighbors
                    for row in grid.grid:
                        for spot in row:
                            spot.update_neighbors(grid.grid)

                    def custom_draw():
                        grid.draw()
                        win.blit(grid_surface, (0, 0))
                        draw_ui(win, [b for b, _ in buttons] + [clear_button, maze_button], selected_algorithm,
                                     stats, font, title_font)
                        pygame.display.update()

                    # Run selected algorithm
                    algorithm_func = buttons[selected_algorithm][1]
                    found = algorithm_func(custom_draw, grid, start, end)
                    stats = {'found': found}
                    sidebar_needs_redraw = True

                if event.key == pygame.K_c:
                    start = None
                    end = None
                    grid.reset()
                    stats = {}
                    sidebar_needs_redraw = True

                if event.key == pygame.K_r:
                    generate_random_maze(grid)

        grid.draw()

        win.blit(grid_surface, (0, 0))

        if sidebar_needs_redraw:
            draw_ui(win, [b for b, _ in buttons] + [clear_button, maze_button], selected_algorithm, stats, font,
                         title_font)
            sidebar_needs_redraw = False

        pygame.display.update()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
