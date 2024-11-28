import copy
class HeuristicAgent:
    def __init__(self):
        return
    def select_action_index(self, state, env):
        """
        Select an action based on heuristic evaluation.
        """
        valid_action_indices = env.get_valid_action_indices()
        if not valid_action_indices:
            return None  # No valid actions

        # Get the actual actions corresponding to the valid indices
        action_mapper = env.game_state.get_action_space_mapper()
        valid_actions = [action_mapper.index_to_action(idx) for idx in valid_action_indices]

        # Evaluate each action using the heuristic function
        action_values = []
        for action in valid_actions:
            value = self.evaluate_action(action, env)
            action_values.append((value, action))

        # Select the action with the highest heuristic value
        best_value, best_action = max(action_values, key=lambda x: x[0])

        # Get the index of the selected action
        selected_index = action_mapper.action_to_index(best_action)

        return selected_index

    def evaluate_action(self, action, env):
        """
        Evaluate the heuristic value of an action.
        """
        factory_idx, tile_color, pattern_line_idx = action
        game_state = env.game_state  # Current game state

        # Copy the game state to simulate the action
        simulated_game_state = copy.deepcopy(game_state)
        player_board = simulated_game_state.player_boards[env.current_player]

        # Simulate taking the action
        try:
            simulated_game_state.take_action(env.current_player, factory_idx, tile_color, pattern_line_idx)
        except ValueError:
            return float('-inf')  # Invalid action

        value = 0

        # Heuristic Factors:

        # 2. Completion Potential
        # Reward completing a pattern line
        if pattern_line_idx != 'floor':
            pattern_line = player_board['pattern_lines'][pattern_line_idx]
            if len(pattern_line) == pattern_line_idx + 1:
                value += 8  # Adjust the reward as needed

        # 3. Avoid Floor Line
        # Penalize actions that add tiles to the floor line
        floor_line_length = len(player_board['floor_line'])
        if floor_line_length > 0:
            value -= floor_line_length * 2  # Adjust the penalty as needed

        # 4. Maximizing Immediate Points
        # Estimate the points that will be gained from placing the tile on the wall
        potential_points = self.estimate_wall_placement_points(env, player_board, pattern_line_idx, tile_color)
        value += potential_points

        # 5. Blocking Opponents (Optional)
        # Not implemented here but can be added by analyzing opponents' boards

        return value

    def estimate_wall_placement_points(self, env, player_board, pattern_line_idx, tile_color):
        """
        Estimate the points gained by placing a tile on the wall.
        """
        if pattern_line_idx == 'floor':
            return 0
        wall = player_board['wall']
        row = wall[pattern_line_idx]
        col_idx = self.get_wall_column_for_tile(env, pattern_line_idx, tile_color)
        if row[col_idx] is not None:
            return 0  # Tile already placed, no points
        # Points for adjacent tiles
        points = 1  # Base point for placing a tile
        # Check horizontal adjacency
        horizontal_count = 0
        left = col_idx - 1
        while left >= 0 and row[left] is not None:
            horizontal_count += 1
            left -= 1
        right = col_idx + 1
        while right < len(row) and row[right] is not None:
            horizontal_count += 1
            right += 1
        if horizontal_count > 0:
            points += horizontal_count

        # Check vertical adjacency
        vertical_count = 0
        up = pattern_line_idx - 1
        while up >= 0 and wall[up][col_idx] is not None:
            vertical_count += 1
            up -= 1
        down = pattern_line_idx + 1
        while down < len(wall) and wall[down][col_idx] is not None:
            vertical_count += 1
            down += 1
        if vertical_count > 0:
            points += vertical_count

        return points

    def get_wall_column_for_tile(self, env, row_idx, tile_color):
        """
        Get the column index on the wall where the tile should be placed.
        """
        # Assuming standard Azul wall pattern
        tile_order = env.game_state.tile_colors  # The standard tile color order
        shift = row_idx
        index = (tile_order.index(tile_color) + shift) % len(tile_order)
        return index
