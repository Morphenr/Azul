import pygame
import sys


class GameVisualiser:
    TILE_SIZE = 40
    TILE_MARGIN = 5
    SCREEN_WIDTH = 1200
    SCREEN_HEIGHT = 800
    COLORS = {
        'red': (255, 0, 0),
        'blue': (0, 0, 255),
        'yellow': (255, 255, 0),
        'black': (0, 0, 0),
        'white': (255, 255, 255),
        None: (200, 200, 200),  # Empty spaces
    }

    def __init__(self):
        """
        Initialize the visualizer without a specific game state.
        """
        pygame.init()
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.display.set_caption("Azul Game Visualizer")
        self.font = pygame.font.Font(None, 30)

    def draw_tile(self, x, y, color, label=None):
        """
        Draw a tile at the specified location.
        """
        rect = pygame.Rect(
            x, y, self.TILE_SIZE - self.TILE_MARGIN, self.TILE_SIZE - self.TILE_MARGIN
        )
        pygame.draw.rect(self.screen, self.COLORS[color], rect)
        if label:
            text = self.font.render(label, True, (255, 255, 255))
            text_rect = text.get_rect(center=rect.center)
            self.screen.blit(text, text_rect)

    def draw_factories(self, game_state):
        """
        Draw the factories and the center pool from the game state.
        """
        factory_radius = self.TILE_SIZE
        start_x = 50
        start_y = 50
        for idx, factory in enumerate(game_state.factories):
            x = start_x + (idx % 3) * 150
            y = start_y + (idx // 3) * 150
            pygame.draw.circle(self.screen, (150, 150, 150), (x, y), factory_radius)
            for i, tile in enumerate(factory):
                tile_x = x - factory_radius + (i % 2) * self.TILE_SIZE
                tile_y = y - factory_radius + (i // 2) * self.TILE_SIZE
                self.draw_tile(tile_x, tile_y, tile)

        # Draw the center pool
        center_x = self.SCREEN_WIDTH // 2
        center_y = self.SCREEN_HEIGHT // 3
        pygame.draw.circle(self.screen, (200, 200, 200), (center_x, center_y), factory_radius)
        for i, tile in enumerate(game_state.center_pool):
            tile_x = center_x - factory_radius + (i % 4) * self.TILE_SIZE
            tile_y = center_y - factory_radius + (i // 4) * self.TILE_SIZE
            self.draw_tile(tile_x, tile_y, tile)

    def draw_player_boards(self, game_state):
        """
        Draw the player boards with pattern lines, walls, floor lines, and scores.
        """
        board_start_x = 50
        board_start_y = 400
        for idx, board in enumerate(game_state.player_boards):
            x = board_start_x + (idx % 2) * 600
            y = board_start_y + (idx // 2) * 200

            # Display the player's score
            score = board["score"]
            score_text = self.font.render(f"Player {idx + 1}: {score} points", True, (0, 0, 0))
            self.screen.blit(score_text, (x, y - 30))

            # Draw pattern lines
            for i, pattern_line in enumerate(board["pattern_lines"]):
                for j in range(game_state.pattern_line_size):
                    tile_color = pattern_line[j] if j < len(pattern_line) else None
                    tile_x = x + j * self.TILE_SIZE
                    tile_y = y + i * self.TILE_SIZE
                    self.draw_tile(tile_x, tile_y, tile_color)

            # Draw the wall
            for i, row in enumerate(board["wall"]):
                for j, tile in enumerate(row):
                    tile_x = x + 300 + j * self.TILE_SIZE
                    tile_y = y + i * self.TILE_SIZE
                    self.draw_tile(tile_x, tile_y, tile)

            # Draw floor line
            for i, tile in enumerate(board["floor_line"]):
                tile_x = x + i * self.TILE_SIZE
                tile_y = y + game_state.pattern_line_size * self.TILE_SIZE + 20
                self.draw_tile(tile_x, tile_y, tile)

    def render(self, game_state):
        """
        Render the Azul game state dynamically.
        :param game_state: The current game state to visualize.
        """
        running = True
        while running:
            self.screen.fill((240, 240, 240))  # Background color
            self.draw_factories(game_state)
            self.draw_player_boards(game_state)
            pygame.display.flip()

            # Event loop
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    print(f"Mouse clicked at {event.pos}")

        pygame.quit()
        sys.exit()
