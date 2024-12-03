import yaml
import math


def get_valid_actions(game_state, player_idx):
    """
    Get all valid actions for the current board state and player, dynamically
    using values from the game state.
    """
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

    return actions


def load_game_settings(settings_path="game/game_settings.yaml"):
    with open(settings_path, 'r') as file:
        settings = yaml.safe_load(file)
    return settings
