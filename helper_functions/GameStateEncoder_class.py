class GameStateEncoder:
    def __init__(self):
        """
        Initialize the encoder. No dependency on the game state during initialization.
        """
        pass

    def encode(self, game_state):
        """
        Encode the game state into a simplified, normalized numerical format.
        Returns a flattened numerical representation.
        """
        self.game_state = game_state  # Store game state for internal use
        self.tile_color_mapping = game_state.tile_color_mapping
        self.num_tile_colors = len(game_state.tile_colors)
        self.num_factories = game_state.num_factories

        features = []

        # Encode factories and center pool with counts of each tile color
        #print(f"Encoded factories: {self._encode_factories()}")
        features.extend(self._encode_factories())
        #print(f"Encoded center pool: {self._encode_center_pool()}")
        features.extend(self._encode_center_pool())

        # Encode first player tile presence
        #print(f"First player tile present: {1 if self._is_first_player_tile_present() else 0}")
        features.append(1 if self._is_first_player_tile_present() else 0)

        # Encode current player board
        #print(f"Encoded player board: {self._encode_player_board(game_state.current_player)}")
        features.extend(self._encode_player_board(game_state.current_player))

        # Encode simplified opponents' boards
        #print(f"Encoded opponents' boards: {self._encode_opponents_boards()}")
        features.extend(self._encode_opponents_boards())

        return features

    def _encode_factories(self):
        """
        Encode the factories as counts of each tile color.
        """
        factory_features = []
        for factory in self.game_state.factories:
            color_counts = [factory.count(color) for color in self.game_state.tile_colors]
            factory_features.extend(color_counts)
        return factory_features

    def _encode_center_pool(self):
        """
        Encode the center pool as counts of each tile color.
        """
        center_color_counts = [self.game_state.center_pool.count(color) for color in self.game_state.tile_colors]
        return center_color_counts

    def _is_first_player_tile_present(self):
        """
        Check if the first player tile is in the center pool.
        """
        return self.game_state.first_player_tile

    def _encode_player_board(self, player_idx):
        """
        Encode the board of the current player, including:
        - Pattern lines: counts of each tile color per line
        - Wall: flattened into a list of tile indices (color indices)
        - Floor line: total length of the floor line
        """
        board = self.game_state.player_boards[player_idx]
        features = []

        # Pattern lines
        for line in board["pattern_lines"]:
            color_counts = [line.count(color) for color in self.game_state.tile_colors]
            features.extend(color_counts)

        # Wall (flattened to tile color indices, None replaced with -1)
        wall_flattened = [
            self.game_state.tile_color_mapping.get(tile, -1)+1 for row in board["wall"] for tile in row # trying to avoid -14 as a value in the vector
        ]
        features.extend(wall_flattened)

        # Floor line (total length)
        features.append(len(board["floor_line"]))

        return features

    def _encode_opponents_boards(self):
        """
        Simplify opponents' board encodings to focus on:
        - The counts of each tile color in the pattern lines
        - The counts of each tile color already placed on the wall
        """
        features = []
        for idx, board in enumerate(self.game_state.player_boards):
            if idx == self.game_state.current_player:
                continue  # Skip the current player

            # Aggregate tile counts in pattern lines
            pattern_color_counts = [0] * len(self.game_state.tile_colors)
            for line in board["pattern_lines"]:
                for color in self.game_state.tile_colors:
                    pattern_color_counts[self.game_state.tile_colors.index(color)] += line.count(color)

            # Aggregate tile counts on the wall
            wall_flattened = [tile for row in board["wall"] for tile in row if tile is not None]
            wall_color_counts = [wall_flattened.count(color) for color in self.game_state.tile_colors]

            # Combine counts for features
            features.extend(pattern_color_counts)
            features.extend(wall_color_counts)

        return features
    def test_encoding(self, game_state):
        """
        Run a test of the encoding function, returning the encoded features for the given game state.
        """
        encoded_features = self.encode(game_state)
        #("Encoded Features:", encoded_features)
        print("Number of Features:", len(encoded_features))
        return encoded_features
