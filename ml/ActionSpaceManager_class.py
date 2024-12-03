class ActionSpaceManager:
    def __init__(self, game_state):
        """
        Initialize the ActionSpaceManager with game-specific parameters.
        """
        self.num_factories = game_state.num_factories
        self.num_colors = len(game_state.tile_colors)
        self.pattern_line_size = game_state.pattern_line_size
        self.tile_color_mapping = game_state.tile_color_mapping
        self.action_space_size = (self.num_factories + 1) * self.num_colors * (self.pattern_line_size + 1)  # +1 for the center and floor line

    def get_valid_actions(self, game_state):
        """
        Get all valid actions for the current board state and player.
        """
        actions = []
        factories = game_state.factories
        center_pool = game_state.center_pool
        player_idx = game_state.current_player
        player_board = game_state.player_boards[player_idx]
        pattern_lines = player_board["pattern_lines"]
        wall = player_board["wall"]
        wall_pattern = player_board["wall_pattern"]

        # Add actions for each factory
        for factory_idx, factory in enumerate(factories):
            tile_colors = set(factory)
            for tile in tile_colors:
                # Check if tile can be placed on a pattern line
                for pattern_line_idx in range(len(pattern_lines)):
                    if (
                        not pattern_lines[pattern_line_idx] or
                        (len(pattern_lines[pattern_line_idx]) < pattern_line_idx + 1 and
                         all(t == tile for t in pattern_lines[pattern_line_idx]))
                    ):
                        # Ensure the tile color is not already in the corresponding wall row
                        if tile not in wall[pattern_line_idx]:
                            actions.append((factory_idx, tile, pattern_line_idx))
                # Add action to place tile on the floor line
                actions.append((factory_idx, tile, "floor"))

        # Add actions for the center pool
        tile_colors = set(center_pool)
        for tile in tile_colors:
            for pattern_line_idx in range(len(pattern_lines)):
                if (
                    not pattern_lines[pattern_line_idx] or
                    (len(pattern_lines[pattern_line_idx]) < pattern_line_idx + 1 and
                     all(t == tile for t in pattern_lines[pattern_line_idx]))
                ):
                    # Ensure the tile color is not already in the corresponding wall row
                    if tile not in wall[pattern_line_idx]:
                        actions.append(("center", tile, pattern_line_idx))
            # Add action to place tile on the floor line
            actions.append(("center", tile, "floor"))

        return actions

    def action_to_index(self, action):
        """
        Convert a given action to a unique index in the action space.
        """
        factory_or_center, tile_color, line = action

        # Determine factory/center index (factories: 0 to num_factories - 1, center: num_factories)
        factory_idx = self.num_factories if factory_or_center == "center" else factory_or_center

        # Determine tile color index
        color_idx = self.tile_color_mapping.get(tile_color, -1)

        # Determine line index (pattern lines: 0 to pattern_line_size - 1, floor: pattern_line_size)
        line_idx = self.pattern_line_size if line == "floor" else line

        # Compute unique index
        return (factory_idx * self.num_colors * (self.pattern_line_size + 1)) + (color_idx * (self.pattern_line_size + 1)) + line_idx

    def index_to_action(self, index):
        """
        Convert a unique index back to an action tuple.
        """
        factory_idx = index // (self.num_colors * (self.pattern_line_size + 1))
        remainder = index % (self.num_colors * (self.pattern_line_size + 1))
        color_idx = remainder // (self.pattern_line_size + 1)
        line_idx = remainder % (self.pattern_line_size + 1)

        # Map back to action components
        factory_or_center = "center" if factory_idx == self.num_factories else factory_idx
        tile_color = list(self.tile_color_mapping.mapping.keys())[color_idx]  # Corrected to use mapping.keys()
        line = "floor" if line_idx == self.pattern_line_size else line_idx

        return (factory_or_center, tile_color, line)

    def get_sorted_actions(self, activations, game_state):
        """
        Translate neural network activations into a sorted array of valid actions based on activation values.
        """
        # Map valid actions to their indices
        valid_actions = self.get_valid_actions(game_state)

        valid_indices = [self.action_to_index(action) for action in valid_actions]

        # Extract the activations for valid actions
        valid_activations = [(activations[idx], idx) for idx in valid_indices]

        # Sort actions by activation values in descending order
        sorted_indices = sorted(valid_activations, key=lambda x: -x[0])
        sorted_actions = [self.index_to_action(idx) for _, idx in sorted_indices]

        return sorted_actions
