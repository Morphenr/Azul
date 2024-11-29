import yaml
import math
    
def get_valid_actions(game_state, player_idx):
    """
    Get all valid actions for the current board state and player, dynamically
    using values from the game state.
    """
    max_actions = game_state.max_actions
    actions = []
    factories = game_state.factories
    center_pool = game_state.center_pool
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

    # Ensure we don't exceed the maximum number of actions
    while len(actions) < max_actions:
        actions.append(None)  # Use None or another placeholder for invalid actions

    # Truncate to max_actions if necessary
    actions = actions[:max_actions]

    return actions


def encode_board_state(game_state, current_player_idx):
    """
    Encode the game state into a simplified format suitable for ML models.
    Returns a flattened numerical representation with reduced features.
    """
    features = []

    tile_color_mapping = game_state.tile_color_mapping
    num_tile_types = len(game_state.tile_colors)
    pattern_line_size  = game_state.pattern_line_size
    factory_size = game_state.factory_size

    # Constants
    num_players = game_state.num_players

    # Encode factories: For each factory, counts of each tile color
    for factory in game_state.factories:
        factory_counts = [0] * num_tile_types
        for tile in factory:
            color_idx = tile_color_mapping.get(tile, -1)
            if color_idx != -1:
                factory_counts[color_idx] += 1
        factory_counts = [count / factory_size for count in factory_counts]
        features.extend(factory_counts)

    # Encode center pool: Counts of each tile color
    center_pool_counts = [0] * num_tile_types
    for tile in game_state.center_pool:
        color_idx = tile_color_mapping.get(tile, -1)
        if color_idx != -1:
            center_pool_counts[color_idx] += 1
    center_pool_counts = [count / (num_tile_types * factory_size) for count in center_pool_counts]
    features.extend(center_pool_counts)

    # Get current player's board
    current_player_board = game_state.player_boards[current_player_idx]

    # Encode current player's pattern lines
    for line_idx, line in enumerate(current_player_board["pattern_lines"]):
        if line:
            # Assuming pattern lines can only have one color per line
            tile_color = tile_color_mapping.get(line[0], 0)
            num_tiles = len(line)
        else:
            tile_color = 0
            num_tiles = 0
        features.append(tile_color / num_tile_types)
        features.append(num_tiles / (line_idx + 1) )

    # Encode current player's wall: counts per tile color
    wall_tile_counts = [0] * num_tile_types
    for row in current_player_board["wall"]:
        for tile in row:
            if tile != 0:
                color_idx = tile_color_mapping.get(tile, -1)
                if color_idx != -1:
                    wall_tile_counts[color_idx] += 1
    features.extend(wall_tile_counts) # not normalised as already [0,1]

    # Encode current player's floor line: counts per tile color
    floor_line_counts = [0] * num_tile_types
    for tile in current_player_board["floor_line"]:
        color_idx = tile_color_mapping.get(tile, -1)
        if color_idx != -1:
            floor_line_counts[color_idx] += 1
    features.extend(floor_line_counts) # not normalised

    # Encode opponent(s) with less detail
    for idx in range(num_players):
        if idx == current_player_idx:
            continue
        opponent_board = game_state.player_boards[idx]

        # Total tiles in pattern lines
        total_pattern_tiles = sum(len(line) for line in opponent_board["pattern_lines"])
        features.append(total_pattern_tiles / (pattern_line_size * 5))

        # Total tiles on wall
        total_wall_tiles = sum(1 for row in opponent_board["wall"] for tile in row if tile != 0)
        features.append(total_wall_tiles / pattern_line_size**2)

        # Total tiles in floor line
        total_floor_tiles = len(opponent_board["floor_line"])

        features.extend([total_pattern_tiles, total_wall_tiles, total_floor_tiles])

    return features




def load_game_settings(settings_path="game/game_settings.yaml"):
    with open(settings_path, 'r') as file:
        settings = yaml.safe_load(file)
    return settings

def calculate_positive_attributes(game_state, player_idx):
    """
    Calculate the positive attributes of a board: pattern line progress, future potential, and end-of-game bonuses.
    """
    player_board = game_state.player_boards[player_idx]
    wall = player_board["wall"]
    pattern_lines = player_board["pattern_lines"]
    
    score = 0

    # Add current game score to evaluation 
    score += player_board["score"]

    # Pattern line progress: Reward pattern lines close to completion
    pattern_progress = sum((len(line) / (idx + 1)) for idx, line in enumerate(pattern_lines))
    score += pattern_progress * 2  # Weighted factor

    # Future potential: Reward rows/columns with fewer empty spaces
    for row_idx, row in enumerate(wall):
        empty_spaces = row.count(None)
        score += (5 - empty_spaces) ** 2  # Reward completed rows more heavily

    # End-of-game bonuses (only if game is over)
    if game_state.is_game_over():
        # Horizontal line bonus: 2 points per complete row
        for row in wall:
            if all(tile is not None for tile in row):
                score += 2

        # Vertical line bonus: 7 points per complete column
        for col_idx in range(5):  # Assuming 5 columns
            if all(wall[row_idx][col_idx] is not None for row_idx in range(5)):
                score += 7

        # Color set bonus: 10 points for each color with all 5 tiles
        color_counts = {}
        for row in wall:
            for tile in row:
                if tile is not None:
                    color_counts[tile] = color_counts.get(tile, 0) + 1
        score += sum(10 for count in color_counts.values() if count == 5)

        # Winning bonus
        winning_score = max(player["score"] for player in game_state.player_boards)
        if player_board["score"] == winning_score:
            score += 100

    return score

def calculate_negative_attributes(game_state, player_idx):
    """
    Calculate the negative attributes of a board: clustering penalty and floor penalties.
    """
    player_board = game_state.player_boards[player_idx]
    wall = player_board["wall"]
    floor_line = player_board["floor_line"]
    
    score = 0

    # Wall density: Penalize even distribution of tiles and reward clusters
    clustering_penalty = calculate_wall_clustering_penalty(wall)
    score -= clustering_penalty

    # Floor penalties: Penalize tiles in the floor line
    score += 0.1 * calculate_floor_penalty(floor_line)

    return score

def evaluate_board_state(game_state, player_idx):
    """
    Provide a holistic evaluation of the player's board state compared to opponents.
    Positive attributes are added, and negative attributes are subtracted for the player.
    The same evaluation is performed for the opponent(s), and the player's score is 
    adjusted by subtracting the opponent's evaluation.
    """
    # Evaluate positive and negative attributes for the player
    player_positive = calculate_positive_attributes(game_state, player_idx)
    player_negative = calculate_negative_attributes(game_state, player_idx)
    player_score = player_positive+ player_negative  # Total score for the player

    # Opponent evaluation
    opponent_scores = []
    for opp_idx, opp_board in enumerate(game_state.player_boards):
        if opp_idx != player_idx:
            # Evaluate positive and negative attributes for each opponent
            opponent_positive = calculate_positive_attributes(game_state, opp_idx)
            opponent_negative = calculate_negative_attributes(game_state, opp_idx)
            opponent_score = opponent_positive + opponent_negative
            opponent_scores.append(opponent_score)

    # Subtract the sum of opponent scores from the player's score (zero-sum game)
    score = player_score - 0.1*(1/len(game_state.player_boards)) * sum(opponent_scores)

    return score


def calculate_wall_clustering_penalty(wall):
    """
    Penalize walls with an even distribution of tiles and reward clustered tiles.
    Measures clustering by counting adjacent tiles and comparing to total tiles.
    A higher clustering score means fewer penalties.
    """
    total_tiles = 0
    adjacency_score = 0

    rows, cols = len(wall), len(wall[0])
    for row in range(rows):
        for col in range(cols):
            if wall[row][col] is not None:
                total_tiles += 1
                # Check adjacent tiles
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Up, down, left, right
                    adj_row, adj_col = row + dr, col + dc
                    if 0 <= adj_row < rows and 0 <= adj_col < cols and wall[adj_row][adj_col] is not None:
                        adjacency_score += 1

    if total_tiles == 0:
        return 0  # No penalty for empty walls

    # Normalize adjacency score by dividing by total possible adjacent pairs
    max_adjacencies = total_tiles * 4  # Each tile can have 4 neighbors
    clustering_score = adjacency_score / max_adjacencies

    # Penalize evenly spread tiles (low clustering score leads to higher penalty)
    return math.exp(-5 * clustering_score) * 10  # Adjust exponential scaling and weight as needed

def calculate_floor_penalty(floor_line):
    """
    Calculate the total penalty for tiles left on the floor line.
    Uses a predefined penalty structure where the first few tiles incur less penalty.
    """
    penalties = [-1, -1, -2, -2, -2, -3, -3]  # Standard Azul floor penalties
    total_penalty = 0

    for idx, tile in enumerate(floor_line):
        if idx < len(penalties):
            total_penalty += penalties[idx]
        else:
            total_penalty += penalties[-1]  # Apply max penalty for additional tiles

    return total_penalty