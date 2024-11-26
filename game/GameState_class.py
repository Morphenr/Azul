import random
from helper_functions.helper_functions import load_game_settings
from helper_functions.TileColorMapping_class import TileColorMapping
from helper_functions.ActionSpaceMapper_class import ActionSpaceMapper

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
        self.pattern_line_size = self.settings.get("pattern_line_size")

        self.tile_color_mapping = TileColorMapping(self.settings["tile_colors"])
        # Initialize the ActionSpaceMapper
        self.action_space_mapper = ActionSpaceMapper(self)
        
        print(f"Loaded {self.num_players} players, {self.num_factories} factories, and tile colors: {self.tile_colors}")

        # Ensure the number of factories and players are valid
        if self.num_factories <= 0:
            raise ValueError("Number of factories must be greater than zero.")
        if self.num_players <= 0:
            raise ValueError("Number of players must be greater than zero.")
    

        # Initialize factories, center pool, and player boards
        self.factories = [[] for _ in range(self.num_factories)]
        self.center_pool = []
        self.player_boards = [
            {
                "pattern_lines": [[] for _ in range(self.pattern_line_size)],
                "wall": [row[:] for row in self.settings.get("wall_pattern")],
                "floor_line": [],
                "score": 0,
                "wall_pattern": self.settings.get("wall_pattern")
            }
            for _ in range(self.num_players)
        ]
        
        self.max_board_size = self.calculate_max_board_size()
        self.max_actions = self.calculate_max_actions()

        self.round_number = 1
        self.bag = self.initialize_bag()
        self.discard_pile = []

        print("GameState initialization complete.")
    
    def __str__(self):
        """
        Converts the game state to a human-readable string format.
        """
        game_state_str = f"Round {self.round_number}\n"
        game_state_str += f"Possible Tile Colors: {self.tile_colors}\n"
        game_state_str += f"Factories: \n"
        
        # Format the factories
        for idx, factory in enumerate(self.factories):
            game_state_str += f"  Factory {idx + 1}: {factory}\n"

        game_state_str += f"Center Pool: {self.center_pool}\n"
        
        # Format the player boards
        for player_idx, board in enumerate(self.player_boards):
            game_state_str += f"Player {player_idx + 1} Board:\n"
            game_state_str += f"  Pattern Lines: {board['pattern_lines']}\n"
            game_state_str += f"  Wall: \n"
            for row in board['wall']:
                game_state_str += f"    {row}\n"
            game_state_str += f"  Floor Line: {board['floor_line']}\n"
            game_state_str += f"  Score: {board['score']}\n"

        game_state_str += f"Discard Pile: {self.discard_pile}\n"
        game_state_str += f"Tiles in Bag: {self.bag}\n"

        return game_state_str


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
                    raise ValueError(f"Both bag and discad pile are empty, cannot draw tiles. Game state: {self.__str__()}")
                # Refill the bag from the discard pile if it's empty
                print("Refilling the tile bag from the discard pile...")
                self.bag = self.discard_pile[:]
                self.discard_pile = []
                random.shuffle(self.bag)
            if self.bag:
                tiles.append(self.bag.pop())
        #print(f"Tiles drawn: {tiles}")
        return tiles

    def refill_factories(self):
        """
        Refill the factories at the start of a new round.
        Raise an error if any factory is non-empty.
        """
        print("Refilling factories for a new round...")
        
        for factory in self.factories:
            if factory:  # Check if the factory is not empty
                raise AssertionError("Cannot refill a non-empty factory.")
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
            board["score"] = 0
        self.refill_factories()
        print("Game state reset complete.")

    def is_round_over(self):
        """
        Check if the round is over, i.e., all factories and the center pool are empty.
        """
        round_over_bool = (all(len(factory) == 0 for factory in self.factories) and len(self.center_pool) == 0)
        return round_over_bool
    
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
            wall_pattern = player_board["wall_pattern"]

            # Score and move tiles for each pattern line
            for i, pattern_line in enumerate(pattern_lines):
                if len(pattern_line) == i + 1:  # Pattern line is full
                    tile_color = pattern_line[0]

                    # Find where to place the tile in the wall
                    row_pattern = wall_pattern[i]  # Get the pattern for the current row
                    column = self.find_wall_column(row_pattern, tile_color)
                    
                    # Check if the wall already has a tile in the column
                    if wall[i][column] is not None:
                        raise ValueError(f"Cannot place {tile_color} in row {i}: spot already occupied.")

                    # Place the tile in the correct position on the wall
                    wall[i][column] = tile_color
                    player_board["score"] += self.calculate_scoring(wall, i, tile_color)  # Custom scoring logic
    
                    self.discard_pile.extend(pattern_line[:-1])  # Leave out the last tile
                    pattern_line.clear()

            # Add floor line penalties
            player_board["score"] -= self.calculate_floor_penalty(floor_line)
            self.discard_pile.extend(floor_line)
            floor_line.clear()

        if self.is_game_over():
            self.apply_end_game_bonuses(player_board, wall)

        # Reset for next round
        self.round_number += 1
        self.refill_factories()

    def apply_end_game_bonuses(self, player_board, wall):
        """
        Apply end-of-game bonuses for completed horizontal and vertical lines
        and full color sets.
        """
        print("Applying end-of-game bonuses...")

        # Horizontal Line Bonus: Check for complete horizontal lines
        for row in wall:
            if None not in row:  # If there are no None values in the row, it's complete
                player_board["score"] += 2  # 2 points for each complete horizontal line

        # Vertical Line Bonus: Check for complete vertical lines
        for col_idx in range(5):  # There are 5 columns in the wall
            if all(wall[row_idx][col_idx] is not None for row_idx in range(5)):  # If column is complete
                player_board["score"] += 7  # 7 points for each complete vertical line

        # Full Color Set Bonus: Check if a color is completed across the entire wall
        for color in ['red', 'blue', 'yellow', 'black', 'white']:  # List of colors
            color_count = sum(1 for row in wall for tile in row if tile == color)
            if color_count == 5:  # All 5 tiles of this color are placed
                player_board["score"] += 10  # 10 points for completing a color

    def find_wall_column(self, wall_row, tile_color):
        """
        Find the appropriate column in the wall for the given tile color.
        
        wall_row: A list representing a row in the wall pattern.
        tile_color: The color of the tile we are trying to place.

        Returns: The index of the column where the tile should go in the given wall row.
        """
        # Find the index of the tile color in the row
        if tile_color in wall_row:
            return wall_row.index(tile_color)
        else:
            raise ValueError(f"Tile color {tile_color} not found in the given wall row.")


    def calculate_scoring(self, wall, row_idx, tile_color):
        """
        Calculate the score for a tile placed on the wall based on adjacency rules.
        
        wall: A list of lists representing the wall of tiles.
        row_idx: The index of the row where the tile is being placed.
        tile_color: The color of the tile being placed.
        
        Returns: The net amount of points gained by placing the tile.
        """
        # Initialize score
        score = 0
        
        # Find the column index where the tile is being placed
        col_idx = self.find_wall_column(wall[row_idx], tile_color)
        
        # Horizontal scoring (left and right adjacency)
        horizontal_score = 1  # The newly placed tile counts as 1 point
        left = col_idx - 1
        right = col_idx + 1
        
        # Check left side for horizontally adjacent tiles
        while left >= 0 and wall[row_idx][left] is not None:
            horizontal_score += 1
            left -= 1
            
        # Check right side for horizontally adjacent tiles
        while right < len(wall[row_idx]) and wall[row_idx][right] is not None:
            horizontal_score += 1
            right += 1
        
        # Vertical scoring (up and down adjacency)
        vertical_score = 1  # The newly placed tile counts as 1 point
        up = row_idx - 1
        down = row_idx + 1
        
        # Check above for vertically adjacent tiles
        while up >= 0 and wall[up][col_idx] is not None:
            vertical_score += 1
            up -= 1
            
        # Check below for vertically adjacent tiles
        while down < len(wall) and wall[down][col_idx] is not None:
            vertical_score += 1
            down += 1
        
        # Total score is the sum of the horizontal and vertical scores
        score = horizontal_score + vertical_score - 1  # Subtract 1 to avoid double-counting the placed tile
        
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
        max_wall_tiles = len(self.player_boards[0]["wall_pattern"]) * len(self.player_boards[0]["wall_pattern"][0]) * self.num_players
        max_floor_line_tiles = 7 * self.num_players

        return max_factory_tiles + max_center_pool_tles + max_pattern_line_tiles + max_wall_tiles + max_floor_line_tiles
    
    def calculate_max_actions(self):
        """
        Estimate the maximum number of valid actions
        """
        max_factory_actions = self.num_factories * len(self.tile_colors) * (self.pattern_line_size + 1)
        max_centre_actions = len(self.tile_colors) * (self.pattern_line_size + 1)

        return max_centre_actions + max_factory_actions
    
    def is_game_over(self):
        """
        Check if the game ends. The game ends when a player completes a row on their wall.
        """
        for board in self.player_boards:
            for row in board["wall"]:
                if all(tile is not None for tile in row):  # Row is complete
                    print("Game is complete!")
                    print(f"Final Games state: {self.__str__}")
                    return True
        return self.round_number > 100  # Safety net if rounds exceed 100
    
    def get_action_space_mapper(self):
        """
        Provide access to the ActionSpaceMapper.
        """
        return self.action_space_mapper