class ActionSpaceMapper:
    def __init__(self, game_state):
        """
        Initialize the action-to-index and index-to-action mappings
        based on the game's configuration.
        """
        self.num_factories = game_state.num_factories
        self.tile_colors = game_state.tile_colors
        self.pattern_line_size = game_state.pattern_line_size

        self.max_factory_actions = self.num_factories * len(self.tile_colors) * (self.pattern_line_size + 1)
        self.total_actions = self.max_factory_actions + len(self.tile_colors) * (self.pattern_line_size + 1) + 1  # +1 for invalid action

        # Precompute mappings
        self.action_to_index_map = {}
        self.index_to_action_map = {}

        # Invalid action (index 0)
        self.action_to_index_map[None] = 0
        self.index_to_action_map[0] = None

        # Factory actions
        index = 1
        for factory_idx in range(self.num_factories):
            for tile_idx, tile_color in enumerate(self.tile_colors):
                for line_idx in range(self.pattern_line_size + 1):
                    action = (factory_idx, tile_color, line_idx if line_idx < self.pattern_line_size else "floor")
                    self.action_to_index_map[action] = index
                    self.index_to_action_map[index] = action
                    index += 1

        # Center actions
        for tile_idx, tile_color in enumerate(self.tile_colors):
            for line_idx in range(self.pattern_line_size + 1):
                action = ("center", tile_color, line_idx if line_idx < self.pattern_line_size else "floor")
                self.action_to_index_map[action] = index
                self.index_to_action_map[index] = action
                index += 1

    def action_to_index(self, action):
        """
        Convert an action (tuple) to an index.
        """
        return self.action_to_index_map.get(action, 0)  # Default to 0 for invalid action

    def index_to_action(self, index):
        """
        Convert an index to an action (tuple).
        """
        return self.index_to_action_map.get(index, None)  # Default to None for invalid index
