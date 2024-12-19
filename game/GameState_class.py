import random
from helper_functions.helper_functions import load_game_settings
from helper_functions.TileColorMapping_class import TileColorMapping

class GameState:
    def __init__(self, settings_path='game_settings.yaml'):
        #print("Loading game settings...")
        self.settings = load_game_settings()
        
        # Ensure settings are loaded correctly
        if 'num_players' not in self.settings or 'num_factories' not in self.settings or 'tile_colors' not in self.settings:
            raise ValueError("Missing essential settings in the configuration.")
        
        self.num_players = self.settings["num_players"]
        self.num_factories = self.settings["num_factories"]
        self.tile_colors = self.settings["tile_colors"]
        self.pattern_line_size = self.settings.get("pattern_line_size")

        self.tile_color_mapping = TileColorMapping(self.settings["tile_colors"])
        
        #print(f"Loaded {self.num_players} players, {self.num_factories} factories, and tile colors: {self.tile_colors}")

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
                "wall": [[None] * self.pattern_line_size for _ in range(self.pattern_line_size)],
                "floor_line": [],
                "score": 0,
                "wall_pattern": self.settings.get("wall_pattern")
            }
            for _ in range(self.num_players)
        ]

        self.round_number = 1
        self.current_player = 0  # Start with the first player
        self.bag = self.initialize_bag()
        self.discard_pile = []
        self.first_player_tile = True  # Indicates if the first player tile is still in the center pool

        #print("GameState initialization complete.")
    
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
        #print("Initializing the tile bag...")
        tile_bag = []
        for color in self.tile_colors:
            tile_bag.extend([color] * 20)  # Add 20 tiles of each color to the bag
        random.shuffle(tile_bag)
        #print(f"Tile bag initialized with {len(tile_bag)} tiles.")
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
                #("Refilling the tile bag from the discard pile...")
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
        #print("Refilling factories for a new round...")
        
        for factory in self.factories:
            if factory:  # Check if the factory is not empty
                raise AssertionError("Cannot refill a non-empty factory.")
            factory.extend(self.draw_tiles(4))  # Each factory gets 4 tiles
        
        #print(f"Refilled {len(self.factories)} factories.")

    def reset(self):
        #print("Resetting the game state...")
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
        self.current_player = 0
        self.first_palyer_tile = True
        #print("Game state reset complete.")

    def is_round_over(self):
        """
        Check if the round is over, i.e., all factories and the center pool are empty.
        """
        round_over_bool = (all(len(factory) == 0 for factory in self.factories) and len(self.center_pool) == 0)
        return round_over_bool

    def wall_tiling_phase(self, is_score_evaluation=False):
        """
        Perform the Wall-tiling phase, including scoring, moving tiles,
        and discarding leftover tiles. Determine the next starting player
        and handle the first player tile appropriately.
        """
        #print("Performing wall tiling phase...")
        next_starting_player = None

        for player_idx, player_board in enumerate(self.player_boards):
            pattern_lines = player_board["pattern_lines"]
            wall = player_board["wall"]
            floor_line = player_board["floor_line"]
            wall_pattern = player_board["wall_pattern"]

            # Identify if the player has the first player tile
            if "first_player_tile" in floor_line:
                next_starting_player = player_idx  # Set as the starting player for the next round

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
            player_board["score"] += self.calculate_floor_penalty(floor_line)

            # Handle first player tile: Place back into center after penalties
            if "first_player_tile" in floor_line:
                floor_line.remove("first_player_tile")

            # Discard the remaining tiles in the floor line
            self.discard_pile.extend(floor_line)
            floor_line.clear()

            if self.is_game_over():
                self.apply_end_game_bonuses(player_board)

        # Reset for the next round
        ####
        if not is_score_evaluation:
            self.round_number += 1
            self.current_player = next_starting_player if next_starting_player is not None else 0
            self.first_player_tile = True  # Reset the first player tile for the next round
            self.refill_factories()

    def apply_end_game_bonuses(self, player_board):
        """
        Apply end-of-game bonuses for completed horizontal and vertical lines
        and full color sets.
        """
        #print("Applying end-of-game bonuses...")

        # Horizontal Line Bonus: Check for complete horizontal lines
        wall = player_board["wall"]
        for row in wall:
            if None not in row:  # If there are no None values in the row, it's complete
                player_board["score"] += 2  # 2 points for each complete horizontal line

        # Vertical Line Bonus: Check for complete vertical lines
        for col_idx in range(5):  # There are 5 columns in the wall
            if all(wall[row_idx][col_idx] is not None for row_idx in range(5)):  # If column is complete
                player_board["score"] += 7  # 7 points for each complete vertical line

        # Full Color Set Bonus: Check if a color is completed across the entire wall
        for color in self.tile_colors:
            color_count = sum(1 for row in wall for tile in row if tile == color)
            if color_count == self.pattern_line_size:
                player_board["score"] += 10

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

        # Initialize scores as 0
        horizontal_score = 0
        vertical_score = 0

        # Horizontal adjacency
        left = col_idx - 1
        while len(wall[row_idx]) > left >= 0 and wall[row_idx][left] is not None:
            horizontal_score += 1
            left -= 1

        right = col_idx + 1
        while 0 <= right < len(wall[row_idx]) and wall[row_idx][right] is not None:
            horizontal_score += 1
            right += 1

        # Vertical adjacency
        up = row_idx - 1
        while len(wall) > up >= 0 and wall[up][col_idx] is not None:
            vertical_score += 1
            up -= 1

        down = row_idx + 1
        while 0 <= down < len(wall) and wall[down][col_idx] is not None:
            vertical_score += 1
            down += 1

        # Total score includes the placed tile itself only once if it's part of a group
        score = max(1, horizontal_score + 1) + max(1, vertical_score + 1) - 1

        return score

    def calculate_floor_penalty(self, floor_line):
        penalties = self.settings.get("floor_line_penalties", [-1, -1, -2, -2, -2, -3, -3])
        penalty = 0
        for i, tile in enumerate(floor_line):
            penalty += penalties[i] if i < len(penalties) else penalties[-1]
        return penalty

    def is_game_over(self):
        """
        Check if the game ends. The game ends when a player completes a row on their wall.
        """
        for board in self.player_boards:
            for row in board["wall"]:
                if all(tile is not None for tile in row):  # Row is complete
                    #print("Game is complete!")
                    #print(f"Final Games state: {self.__str__()}")
                    return True
        return self.round_number > 100  # Safety net if rounds exceed 100

    def take_action(self, player_idx, factory_idx, tile, pattern_line_idx):
        """
        Allow a player to take an action in the game by selecting tiles from a factory or the center pool,
        and placing them in the specified pattern line or floor line.
        """
        if player_idx != self.current_player:
            raise ValueError("It's not this player's turn.")

        if factory_idx == "center":  # Action from the center pool
            selected_tiles = [t for t in self.center_pool if t == tile]
            if not selected_tiles:
                raise ValueError("Tile not available in center pool.")

            for _ in selected_tiles:
                self.center_pool.remove(tile)

            if self.first_player_tile:  # If this is the first player taking from the center pool
                self.player_boards[player_idx]["floor_line"].append("first_player_tile")
                self.first_player_tile = False  # Mark the first player tile as taken

            if pattern_line_idx == "floor":
                self.player_boards[player_idx]["floor_line"].extend(selected_tiles)
            else:
                pattern_line = self.player_boards[player_idx]["pattern_lines"][pattern_line_idx]
                max_capacity = pattern_line_idx + 1

                while selected_tiles and len(pattern_line) < max_capacity:
                    pattern_line.append(selected_tiles.pop())

                self.player_boards[player_idx]["floor_line"].extend(selected_tiles)

        elif isinstance(factory_idx, int) and 0 <= factory_idx < len(self.factories):  # Valid factory index
            factory = self.factories[factory_idx]
            selected_tiles = [t for t in factory if t == tile]
            if not selected_tiles:
                raise ValueError("Tile not available in the selected factory.")

            for _ in selected_tiles:
                factory.remove(tile)

            self.center_pool.extend(factory)
            factory.clear()

            if pattern_line_idx == "floor":
                self.player_boards[player_idx]["floor_line"].extend(selected_tiles)
            else:
                pattern_line = self.player_boards[player_idx]["pattern_lines"][pattern_line_idx]
                max_capacity = pattern_line_idx + 1

                while selected_tiles and len(pattern_line) < max_capacity:
                    pattern_line.append(selected_tiles.pop())

                self.player_boards[player_idx]["floor_line"].extend(selected_tiles)
        else:
            raise ValueError("Invalid action. Either factory or center pool should be selected.")

        self.current_player = (self.current_player + 1) % self.num_players

        if self.is_round_over():
            self.wall_tiling_phase()

