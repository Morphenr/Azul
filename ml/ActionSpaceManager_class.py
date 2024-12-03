class ActionSpaceManager:
    def __init__(self, game_state):
        """
        Initialize the ActionSpaceManager with game-specific parameters.
        """
        self.num_factories = game_state.num_factories
        self.num_colors = len(game_state.tile_colors)
        self.pattern_line_size = game_state.pattern_line_size
        self.tile_color_mapping = game_state.tile_color_mapping
        self.action_space_size = (self.num_factories + 1) + self.num_colors + (self.pattern_line_size + 1)  # +1 for the center and floor line

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

        #print(f"Current Player: {player_idx} Valid actions: {actions}")
        return actions

    def action_to_index(self, action):
        """
        Convert an action into a tuple of three indices (factory_or_center, tile_color, pattern_line).
        """
        factory_or_center, tile_color, line = action

        # Factory or center index
        factory_idx = self.num_factories if factory_or_center == "center" else factory_or_center

        # Tile color index
        color_idx = self.tile_color_mapping.get(tile_color, -1)

        # Pattern line index
        line_idx = self.pattern_line_size if line == "floor" else line

        return factory_idx, color_idx, line_idx

    def index_to_action(self, indices):
        """
        Convert a tuple of indices (factory_or_center, tile_color, pattern_line) back to an action.
        """
        factory_idx, color_idx, line_idx = indices

        # Map back to action components
        factory_or_center = "center" if factory_idx == self.num_factories else factory_idx
        tile_color = list(self.tile_color_mapping.keys())[color_idx]
        line = "floor" if line_idx == self.pattern_line_size else line_idx

        return factory_or_center, tile_color, line

    def get_sorted_actions(self, activations, game_state):
        """
        Translate neural network activations into a sorted array of valid actions based on activation values.
        """
        # Get valid actions
        valid_actions = self.get_valid_actions(game_state)

        # Map activations to valid actions
        action_scores = []
        for action in valid_actions:
            indices = self.action_to_index(action)
            score = (
                    activations[indices[0]] +  # Factory/center activation
                    activations[self.num_factories + 1 + indices[1]] +  # Tile color activation
                    activations[self.num_factories + 1 + self.num_colors + indices[2]]  # Pattern line activation
            )
            action_scores.append((action, score))

        # Sort valid actions by score
        sorted_actions = sorted(action_scores, key=lambda x: -x[1])

        # Return sorted actions
        return [action for action, _ in sorted_actions]