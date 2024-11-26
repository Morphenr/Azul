import random
from helper_functions.helper_functions import load_game_settings
from helper_functions.TileColorMapping_class import TileColorMapping

class GameState:
    def __init__(self, settings_path='game_settings.yaml'):
        print("Loading game settings...")
        self.settings = load_game_settings()
        
        # Ensure settings are loaded correctly
        if 'num_players' not in self.settings or 'num_factories' not in self.settings or 'tile_colors' not in self.settings:
            raise ValueError("Missing essential settings in the configuration.")
        
        self.num_players = self.settings["num_players"]
        self.num_factories = self.settings["num_factories"]
        self.tile_colors = self.settings["tile_colors"]
        self.tile_color_mapping = TileColorMapping(self.settings["tile_colors"])
        
        print(f"Loaded {self.num_players} players, {self.num_factories} factories, and tile colors: {self.tile_colors}")

        # Ensure the number of factories and players are valid
        if self.num_factories <= 0:
            raise ValueError("Number of factories must be greater than zero.")
        if self.num_players <= 0:
            raise ValueError("Number of players must be greater than zero.")
        
        # Define pattern line size and wall pattern based on the settings (ensure defaults are set)
        self.pattern_line_size = self.settings.get("pattern_line_size", 5)
        self.wall_pattern = self.settings.get("wall_pattern", [[None] * 5 for _ in range(5)])

        # Initialize factories, center pool, and player boards
        self.factories = [[] for _ in range(self.num_factories)]
        self.center_pool = []
        self.player_boards = [
            {
                "pattern_lines": [[] for _ in range(self.pattern_line_size)],
                "wall": [row[:] for row in self.wall_pattern],
                "floor_line": [],
                "score": 0
            }
            for _ in range(self.num_players)
        ]
        
        self.max_board_size = self.calculate_max_board_size()
        self.max_actions = self.calculate_max_actions()

        self.round_number = 1
        self.bag = self.initialize_bag()
        self.discard_pile = []

        print("GameState initialization complete.")
    
    def initialize_bag(self):
        """
        Initialize the tile bag based on the colors defined in the game settings.
        The number of tiles for each color is fixed to 20 for simplicity.
        """
        print("Initializing the tile bag...")
        tile_bag = []
        for color in self.tile_colors:
            tile_bag.extend([color] * 20)  # Add 20 tiles of each color to the bag
        random.shuffle(tile_bag)
        print(f"Tile bag initialized with {len(tile_bag)} tiles.")
        return tile_bag

    def draw_tiles(self, count):
        """
        Draw tiles from the bag. If the bag is empty, refill it from the discard pile.
        """
        tiles = []
        for _ in range(count):
            if not self.bag:
                if not self.discard_pile:
                    raise ValueError("Both bag and discad pile are empty, cannot draw tiles.")
                # Refill the bag from the discard pile if it's empty
                print("Refilling the tile bag from the discard pile...")
                self.bag = self.discard_pile[:]
                self.discard_pile = []
                random.shuffle(self.bag)
            if self.bag:
                tiles.append(self.bag.pop())
        print(f"Tiles drawn: {tiles}")
        return tiles

    def refill_factories(self):
        """
        Refill the factories at the start of a new round.
        """
        print("Refilling factories for a new round...")
        for factory in self.factories:
            factory.clear()  # Clear factories from the previous round
            factory.extend(self.draw_tiles(4))  # Each factory gets 4 tiles
        print(f"Refilled {len(self.factories)} factories.")

    def reset(self):
        print("Resetting the game state...")
        self.round_number = 1
        self.bag = self.initialize_bag()
        self.discard_pile = []
        self.factories = [[] for _ in range(self.num_factories)]
        self.center_pool = []
        for board in self.player_boards:
            board["pattern_lines"] = [[] for _ in range(self.pattern_line_size)]
            board["wall"] = [[None] * 5 for _ in range(5)]  # Reset to default empty wall
            board["floor_line"] = []
        self.refill_factories()
        print("Game state reset complete.")

    def is_round_over(self):
        """
        Check if the round is over, i.e., all factories and the center pool are empty.
        """
        return all(len(factory) == 0 for factory in self.factories) and len(self.center_pool) == 0
    
    def wall_tiling_phase(self):
        """
        Perform the Wall-tiling phase, including scoring, moving tiles,
        and discarding leftover tiles.
        """
        print("Performing wall tiling phase...")
        for player_board in self.player_boards:
            pattern_lines = player_board["pattern_lines"]
            wall = player_board["wall"]
            floor_line = player_board["floor_line"]

            # Score and move tiles for each pattern line
            for i, pattern_line in enumerate(pattern_lines):
                if len(pattern_line) == i + 1:  # Pattern line is full
                    tile_color = pattern_line[0]
                    wall[i][self.find_wall_column(wall[i], tile_color)] = tile_color
                    player_board["score"] += self.calculate_scoring(wall, i, tile_color)  # Custom scoring logic
                    pattern_lines[i] = []  # Clear pattern line

            # Add floor line penalties
            player_board["score"] -= self.calculate_floor_penalty(floor_line)
            floor_line.clear()

            # Discard leftover tiles
            for pattern_line in pattern_lines:
                self.discard_pile.extend(pattern_line)
                pattern_line.clear()

        # Reset for next round
        self.refill_factories()

    def find_wall_column(self, wall_row, tile_color):
        """
        Find the appropriate column in the wall for the given tile color.
        """
        wall_colors = ["blue", "yellow", "red", "black", "white"]  # Example Azul colors
        return wall_colors.index(tile_color)

    def calculate_scoring(self, wall, row_idx, tile_color):
        """
        Calculate the score for a tile placed on the wall.
        """
        score = 1
        # Horizontal scoring
        left = right = row_idx
        while left > 0 and wall[row_idx][left - 1] is not None:
            score += 1
            left -= 1
        while right < len(wall[row_idx]) - 1 and wall[row_idx][right + 1] is not None:
            score += 1
            right += 1
        # Vertical scoring
        col = self.find_wall_column(wall[row_idx], tile_color)
        up = down = row_idx
        while up > 0 and wall[up - 1][col] is not None:
            score += 1
            up -= 1
        while down < len(wall) - 1 and wall[down + 1][col] is not None:
            score += 1
            down += 1
        return score

    def calculate_floor_penalty(self, floor_line):
        """
        Calculate penalties for leftover tiles on the floor line.
        """
        penalties = [-1, -1, -2, -2, -2, -3, -3]  # Example penalty values
        penalty = 0
        for i, tile in enumerate(floor_line):
            penalty += penalties[i] if i < len(penalties) else -3
        return penalty

    def calculate_max_board_size(self):
        """
        Estimate the maximum size of the encoded board state
        """
        max_factory_tiles = self.num_factories * 4

        max_center_pool_tles = max_factory_tiles

        max_pattern_line_tiles = self.pattern_line_size * self.num_players
        max_wall_tiles = len(self.wall_pattern) * len(self.wall_pattern[0]) * self.num_players
        max_floor_line_tiles = 7 * self.num_players

        return max_factory_tiles + max_center_pool_tles + max_pattern_line_tiles + max_wall_tiles + max_floor_line_tiles
    
    def calculate_max_actions(self):
        """
        Estimate the maximum number of valid actions
        """
        max_factory_actions = self.num_factories * len(self.tile_colors) * (self.pattern_line_size + 1)
        max_centre_actions = len(self.tile_colors) * (self.pattern_line_size + 1)

        return max_centre_actions + max_factory_actions