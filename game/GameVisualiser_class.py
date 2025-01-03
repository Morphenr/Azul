import pygame
import sys
import math


class GameVisualiser:
    TILE_SIZE = 40
    TILE_MARGIN = 5
    BASE_SCREEN_WIDTH = 1200
    BASE_SCREEN_HEIGHT = 1000
    COLORS = {
        'red': (255, 100, 100),
        'blue': (100, 100, 255),
        'yellow': (255, 255, 100),
        'black': (50, 50, 50),
        'white': (250, 250, 250),
        'first_player_tile': (0, 200, 100),  # Unique colour for first player tile
        None: (200, 200, 200),  # Empty spaces
    }

    MUTED_COLORS = {
        'red': (255, 230, 230),
        'blue': (230, 230, 255),
        'yellow': (255, 255, 230),
        'black': (200, 200, 200),
        'white': (250, 250, 250),
    }

    def __init__(self, num_players):
        if num_players < 2 or num_players > 4:
            raise ValueError("The number of players must be between 2 and 4.")
        self.num_players = num_players

        self.SCREEN_WIDTH = self.BASE_SCREEN_WIDTH
        self.SCREEN_HEIGHT = self.BASE_SCREEN_HEIGHT + (self.num_players - 2) * 200

        pygame.init()
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.display.set_caption("Azul Game Visualiser")
        self.font = pygame.font.Font(None, 30)
        self.running = True

    def draw_tile(self, x, y, color, label=None):
        rect = pygame.Rect(
            x, y, self.TILE_SIZE - self.TILE_MARGIN, self.TILE_SIZE - self.TILE_MARGIN
        )
        pygame.draw.rect(self.screen, self.COLORS[color], rect)
        pygame.draw.rect(self.screen, (0, 0, 0), rect, 2)
        if label:
            text = self.font.render(label, True, (255, 255, 255))
            text_rect = text.get_rect(center=rect.center)
            self.screen.blit(text, text_rect)

    def draw_muted_tile(self, x, y, color):
        rect = pygame.Rect(
            x, y, self.TILE_SIZE - self.TILE_MARGIN, self.TILE_SIZE - self.TILE_MARGIN
        )
        pygame.draw.rect(self.screen, self.MUTED_COLORS[color], rect)
        pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)

    def draw_factories(self, game_state):
        center_x = self.SCREEN_WIDTH // 2
        center_y = self.SCREEN_HEIGHT // 4
        factory_radius = 200
        angle_step = 2 * math.pi / len(game_state.factories)

        for idx, factory in enumerate(game_state.factories):
            angle = idx * angle_step
            x = center_x + int(factory_radius * math.cos(angle))
            y = center_y + int(factory_radius * math.sin(angle))

            pygame.draw.circle(self.screen, (150, 150, 150), (x, y), 40)
            for i, tile in enumerate(factory):
                tile_x = x - 20 + (i % 2) * self.TILE_SIZE
                tile_y = y - 20 + (i // 2) * self.TILE_SIZE
                self.draw_tile(tile_x, tile_y, tile)

        pygame.draw.circle(self.screen, (200, 200, 200), (center_x, center_y), 50)
        for i, tile in enumerate(game_state.center_pool):
            tile_x = center_x - 40 + (i % 4) * self.TILE_SIZE
            tile_y = center_y - 40 + (i // 4) * self.TILE_SIZE
            self.draw_tile(tile_x, tile_y, tile)

        if game_state.first_player_tile:
            self.draw_tile(center_x - 60, center_y, "first_player_tile", "1")

    def draw_player_boards(self, game_state):
        board_start_x = 50
        board_start_y = 400
        board_spacing_x = 600
        board_spacing_y = 400

        for idx, board in enumerate(game_state.player_boards):
            x = board_start_x + (idx % 2) * board_spacing_x
            y = board_start_y + (idx // 2) * board_spacing_y

            score = board["score"]
            score_text = self.font.render(f"Player {idx + 1}: {score} points", True, (0, 0, 0))
            self.screen.blit(score_text, (x, y - 30))

            for i, pattern_line in enumerate(board["pattern_lines"]):
                for j in range(i + 1):
                    tile_color = pattern_line[j] if j < len(pattern_line) else None
                    tile_x = x + j * self.TILE_SIZE
                    tile_y = y + i * self.TILE_SIZE
                    self.draw_tile(tile_x, tile_y, tile_color)

            wall_start_x = x + 250
            for i, row in enumerate(board["wall"]):
                wall_pattern = board["wall_pattern"][i]
                for j, tile in enumerate(row):
                    tile_x = wall_start_x + j * self.TILE_SIZE
                    tile_y = y + i * self.TILE_SIZE
                    if tile is None:
                        self.draw_muted_tile(tile_x, tile_y, wall_pattern[j])
                    else:
                        self.draw_tile(tile_x, tile_y, tile)

            floor_line_start_y = y + 6 * self.TILE_SIZE + 10
            for i, tile in enumerate(board["floor_line"]):
                tile_x = x + i * self.TILE_SIZE
                self.draw_tile(tile_x, floor_line_start_y, tile)

            pygame.draw.rect(
                self.screen,
                (0, 0, 0),
                (x - 10, y - 40, 450, 7 * self.TILE_SIZE + 60),
                2,
            )

    def render(self, game_state):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

        if not self.running:
            return

        self.screen.fill((240, 240, 240))
        self.draw_factories(game_state)
        self.draw_player_boards(game_state)
        pygame.display.flip()

    def close(self):
        pygame.quit()
