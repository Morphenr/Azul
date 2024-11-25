import yaml

def is_game_over(game_state):
    """
    Check if the game ends. The game ends when a player completes a row on their wall.
    """
    for board in game_state.player_boards:
        for row in board["wall"]:
            if all(tile is not None for tile in row):  # Row is complete
                return True
    return game_state.round_number > 10  # Safety net if rounds exceed 5



def simulate_action(game_state, player_idx, factory_idx, tile, pattern_line_idx):
    if factory_idx < len(game_state.factories):
        factory = game_state.factories[factory_idx]
        if tile in factory:
            game_state.factories[factory_idx].remove(tile)
            if pattern_line_idx == "floor":
                game_state.player_boards[player_idx]["floor_line"].append(tile)
            else:
                game_state.player_boards[player_idx]["pattern_lines"][pattern_line_idx].append(tile)
        else:
            raise ValueError("Invalid tile selection.")
    else:
        raise ValueError("Invalid factory selection.")
    
def get_valid_actions(game_state, player_idx):
    """
    Get all valid actions for the current board state and player.
    """
    actions = []
    factories = game_state.factories
    center_pool = game_state.center_pool
    player_board = game_state.player_boards[player_idx]
    pattern_lines = player_board["pattern_lines"]

    # Log game state for debugging
    print(f"Factories: {factories}")
    print(f"Center Pool: {center_pool}")
    print(f"Player {player_idx} Pattern Lines: {pattern_lines}")

    # Add actions for each factory
    for factory_idx, factory in enumerate(factories):
        tile_colors = set(factory)
        for tile in tile_colors:
            # Check if tile can be placed on a pattern line
            for pattern_line_idx in range(len(pattern_lines)):
                if (not pattern_lines[pattern_line_idx] or
                    (len(pattern_lines[pattern_line_idx]) < pattern_line_idx + 1 and
                     all(t == tile for t in pattern_lines[pattern_line_idx]))):
                    actions.append((factory_idx, tile, pattern_line_idx))
            # Add action to place tile on the floor line
            actions.append((factory_idx, tile, "floor"))

    # Add actions for the center pool
    tile_colors = set(center_pool)
    for tile in tile_colors:
        for pattern_line_idx in range(len(pattern_lines)):
            if (not pattern_lines[pattern_line_idx] or
                (len(pattern_lines[pattern_line_idx]) < pattern_line_idx + 1 and
                 all(t == tile for t in pattern_lines[pattern_line_idx]))):
                actions.append(("center", tile, pattern_line_idx))
        # Add action to place tile on the floor line
        actions.append(("center", tile, "floor"))

    # Log valid actions
    print(f"Valid Actions for Player {player_idx}: {actions}")

    return actions



def encode_board_state(game_state):
    """
    Encode the game state into a format suitable for ML models.
    Returns a flattened numerical representation.
    """
    features = []
    
    # Use the TileColorMapping object from the game state
    tile_color_mapping = game_state.tile_color_mapping
    
    # Encode factories
    for factory in game_state.factories:
        features.extend([tile_color_mapping.get(tile) for tile in factory])  # Map tile colors to integers

    # Encode center pool
    features.extend([tile_color_mapping.get(tile) for tile in game_state.center_pool])  # Map tile colors to integers

    # Encode player boards
    for board in game_state.player_boards:
        # Pattern lines
        for line in board["pattern_lines"]:
            features.extend([tile_color_mapping.get(tile) for tile in line])  # Map tile colors to integers
        # Wall rows
        for wall_row in board["wall"]:
            features.extend([tile_color_mapping.get(tile) for tile in wall_row])  # Map tile colors to integers
        # Floor line
        features.extend([tile_color_mapping.get(tile) for tile in board["floor_line"]])  # Map tile colors to integers

    return features




def calculate_scores(player_board):
    """
    Calculate the immediate score for a player based on the Azul rules.
    """
    wall = player_board["wall"]
    pattern_lines = player_board["pattern_lines"]
    floor_line = player_board["floor_line"]
    score = 0

    # Process pattern lines and move completed rows to the wall
    for row_idx, line in enumerate(pattern_lines):
        if len(line) == row_idx + 1:  # Row is complete
            color = line[0]
            col_idx = row_idx  # Simplified placement
            wall[row_idx][col_idx] = color
            pattern_lines[row_idx] = []

            # Score for the placement
            score += calculate_wall_score(wall, row_idx, col_idx)

    # Floor line penalty
    score -= calculate_floor_penalty(floor_line)
    return score


def calculate_wall_score(wall, row, col):
    """
    Calculate points for placing a tile based on adjacency.
    """
    horizontal = sum(1 for c in range(5) if wall[row][c] is not None) - 1
    vertical = sum(1 for r in range(5) if wall[r][col] is not None) - 1
    return max(1, horizontal + vertical)


def calculate_floor_penalty(floor_line):
    """
    Calculate penalty for tiles in the floor line.
    """
    penalties = [-1, -1, -2, -2, -2, -3, -3]
    return sum(penalties[:len(floor_line)])

def load_game_settings(settings_path="game/game_settings.yaml"):
    with open(settings_path, 'r') as file:
        settings = yaml.safe_load(file)
    return settings