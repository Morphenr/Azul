import pygame
import sys


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
        """
        Initialize the visualiser with dynamic screen size.
        :param num_players: Number of players (2–4).
        """
        if num_players < 2 or num_players > 4:
            raise ValueError("The number of players must be between 2 and 4.")
        self.num_players = num_players

        # Dynamically adjust screen size based on the number of players
        self.SCREEN_WIDTH = self.BASE_SCREEN_WIDTH
        self.SCREEN_HEIGHT = self.BASE_SCREEN_HEIGHT + (self.num_players - 2) * 200

        pygame.init()
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.display.set_caption("Azul Game Visualiser")
        self.font = pygame.font.Font(None, 30)
        self.running = True

    def draw_tile(self, x, y, color, label=None):
        """
        Draw a tile at the specified location.
        """
        rect = pygame.Rect(
            x, y, self.TILE_SIZE - self.TILE_MARGIN, self.TILE_SIZE - self.TILE_MARGIN
        )
        pygame.draw.rect(self.screen, self.COLORS[color], rect)
        pygame.draw.rect(self.screen, (0, 0, 0), rect, 2)  # Add border to tiles
        if label:
            text = self.font.render(label, True, (255, 255, 255))
            text_rect = text.get_rect(center=rect.center)
            self.screen.blit(text, text_rect)

    def draw_muted_tile(self, x, y, color):
        """
        Draw a muted tile (e.g., for the wall pattern).
        """
        rect = pygame.Rect(
            x, y, self.TILE_SIZE - self.TILE_MARGIN, self.TILE_SIZE - self.TILE_MARGIN
        )
        pygame.draw.rect(self.screen, self.MUTED_COLORS[color], rect)
        pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)  # Add border to muted tiles

    def draw_factories(self, game_state):
        """
        Draw the factories and the center pool from the game state.
        """
        factory_radius = self.TILE_SIZE
        start_x = 100
        start_y = 50
        factory_spacing = 200  # Increased spacing between factories

        for idx, factory in enumerate(game_state.factories):
            x = start_x + (idx % 3) * factory_spacing
            y = start_y + (idx // 3) * factory_spacing
            pygame.draw.circle(self.screen, (150, 150, 150), (x, y), factory_radius)
            for i, tile in enumerate(factory):
                tile_x = x - factory_radius + (i % 2) * self.TILE_SIZE
                tile_y = y - factory_radius + (i // 2) * self.TILE_SIZE
                self.draw_tile(tile_x, tile_y, tile)

        # Draw the center pool
        center_x = self.SCREEN_WIDTH // 2
        center_y = self.SCREEN_HEIGHT // 4
        pygame.draw.circle(self.screen, (200, 200, 200), (center_x, center_y), factory_radius)
        for i, tile in enumerate(game_state.center_pool):
            tile_x = center_x - factory_radius + (i % 4) * self.TILE_SIZE
            tile_y = center_y - factory_radius + (i // 4) * self.TILE_SIZE
            self.draw_tile(tile_x, tile_y, tile)

        # Draw the first player tile in the center
        if game_state.first_player_tile:
            self.draw_tile(center_x - factory_radius, center_y, "first_player_tile", "1")

    def draw_player_boards(self, game_state):
        """
        Draw the player boards with pattern lines, walls, floor lines, and scores.
        """
        board_start_x = 50
        board_start_y = 400
        board_spacing_x = 600  # Adjusted horizontal spacing
        board_spacing_y = 400  # Adjusted vertical spacing

        for idx, board in enumerate(game_state.player_boards):
            x = board_start_x + (idx % 2) * board_spacing_x
            y = board_start_y + (idx // 2) * board_spacing_y

            # Display the player's score
            score = board["score"]
            score_text = self.font.render(f"Player {idx + 1}: {score} points", True, (0, 0, 0))
            self.screen.blit(score_text, (x, y - 30))

            # Draw pattern lines
            for i, pattern_line in enumerate(board["pattern_lines"]):
                for j in range(i + 1):  # Incremental rows
                    tile_color = pattern_line[j] if j < len(pattern_line) else None
                    tile_x = x + j * self.TILE_SIZE
                    tile_y = y + i * self.TILE_SIZE
                    self.draw_tile(tile_x, tile_y, tile_color)

            # Draw the wall with muted colours
            wall_start_x = x + 250
            for i, row in enumerate(board["wall"]):
                wall_pattern = board["wall_pattern"][i]
                for j, tile in enumerate(row):
                    tile_x = wall_start_x + j * self.TILE_SIZE
                    tile_y = y + i * self.TILE_SIZE
                    if tile is None:  # Show muted wall pattern if no tile is placed
                        self.draw_muted_tile(tile_x, tile_y, wall_pattern[j])
                    else:
                        self.draw_tile(tile_x, tile_y, tile)

            # Draw floor line
            floor_line_start_y = y + 6 * self.TILE_SIZE + 10
            for i, tile in enumerate(board["floor_line"]):
                tile_x = x + i * self.TILE_SIZE
                self.draw_tile(tile_x, floor_line_start_y, tile)

            # Draw a border around the player board for clarity
            pygame.draw.rect(
                self.screen,
                (0, 0, 0),
                (x - 10, y - 40, 450, 7 * self.TILE_SIZE + 60),
                2,
            )

    def render(self, game_state):
        """
        Render the Azul game state dynamically.
        :param game_state: The current game state to visualize.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

        if not self.running:
            return

        self.screen.fill((240, 240, 240))  # Neutral light grey background
        self.draw_factories(game_state)
        self.draw_player_boards(game_state)
        pygame.display.flip()

    def close(self):
        """
        Properly close the visualiser.
        """
        pygame.quit()
