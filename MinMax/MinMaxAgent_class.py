import math
import time
from copy import deepcopy
from helper_functions.helper_functions import get_valid_actions

class AzulAgent:
    def __init__(self, player_idx, depth=3):
        self.player_idx = player_idx
        self.depth = depth
        self.start_time = None
        self.time_limit = 10  # Default time limit in seconds

    def evaluate(self, game_state, player_idx):
        """
        Evaluate the game state for the given player.
        The score considers the player's current score and penalties.
        It performs a pseudo wall tiling phase to estimate potential points.
        """
        pseudo_game_state = deepcopy(game_state)
        pseudo_game_state.wall_tiling_phase(is_score_evaluation=True)
        board = pseudo_game_state.player_boards[player_idx]
        score = board["score"]
        return score

    def minimax(self, game_state, depth, alpha, beta, maximizing_player):
        """
        Pruned Minimax implementation to decide the best move.
        """
        if time.time() - self.start_time >= self.time_limit:
            return self.evaluate(game_state, self.player_idx), None

        if depth == 0 or game_state.is_game_over():
            return self.evaluate(game_state, self.player_idx), None

        valid_actions = get_valid_actions(game_state, game_state.current_player)
        leaves_visited = 0

        if maximizing_player:
            max_eval = -math.inf
            best_action = None
            for action in valid_actions:
                factory_idx, tile, pattern_line_idx = action
                new_state = deepcopy(game_state)
                new_state.take_action(self.player_idx, factory_idx, tile, pattern_line_idx)
                eval_value, _ = self.minimax(new_state, depth - 1, alpha, beta, False)
                leaves_visited += 1
                if eval_value > max_eval:
                    max_eval = eval_value
                    best_action = action
                alpha = max(alpha, eval_value)
                if beta <= alpha:
                    break
            #print(f"Depth: {self.depth - depth}, Leaves Visited: {leaves_visited}, Best Eval: {max_eval}")
            return max_eval, best_action

        else:
            min_eval = math.inf
            for action in valid_actions:
                factory_idx, tile, pattern_line_idx = action
                new_state = deepcopy(game_state)
                new_state.take_action((self.player_idx + 1) % game_state.num_players, factory_idx, tile, pattern_line_idx)
                eval_value, _ = self.minimax(new_state, depth - 1, alpha, beta, True)
                leaves_visited += 1
                min_eval = min(min_eval, eval_value)
                beta = min(beta, eval_value)
                if beta <= alpha:
                    break
            #print(f"Depth: {self.depth - depth}, Leaves Visited: {leaves_visited}, Min Eval: {min_eval}")
            return min_eval, None

    def iterative_deepening(self, game_state):
        """
        Perform iterative deepening to find the best action within the time limit.
        """
        depth = 1
        best_action = None
        best_value = -math.inf

        while time.time() - self.start_time < self.time_limit:
            print(f"Exploring depth: {depth}")
            try:
                value, action = self.minimax(game_state, depth, -math.inf, math.inf, True)
                if value > best_value:
                    best_value = value
                    best_action = action
                depth += 1
            except Exception as e:
                print(f"Depth {depth} aborted: {e}")
                break

        print(f"Max Depth Reached: {depth - 1}, Best Value: {best_value}, Best Action: {best_action}")
        return best_action

    def choose_action(self, game_state):
        self.start_time = time.time()
        return self.iterative_deepening(game_state)