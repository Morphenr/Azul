import time
import multiprocessing as mp
from copy import deepcopy
from helper_functions.helper_functions import get_valid_actions

class MinMaxAgent:
    def __init__(self, max_depth):
        """
        Initialises the MinMaxAgent with the specified search depth.

        Args:
            max_depth (int): The maximum depth for the Minimax search.
        """
        self.max_depth = max_depth
        self.time_limit = 1.0  # Time limit in seconds
        self.start_time = None
        self.transposition_table = {}  # Cache for previously evaluated game states
        self.nodes_explored = 0
        self.leaf_nodes = 0
        self.achieved_depth = 0

    def evaluate_game_state(self, game_state, player_idx):
        """
        Evaluate the game state by pretending the current move was the last of the round,
        and calculating the resulting score after the wall tiling phase.

        Args:
            game_state (GameState): The current game state.
            player_idx (int): The index of the player whose score is being evaluated.

        Returns:
            int: The score of the game state for the given player minus the average score of opponents.
        """
        self.leaf_nodes += 1
        # Clone the game state to avoid modifying the actual state
        game_state_clone = deepcopy(game_state)

        # Perform the wall tiling phase on the cloned state
        game_state_clone.wall_tiling_phase()

        # Calculate scores
        player_score = game_state_clone.player_boards[player_idx]["score"]
        opponent_scores = [
            game_state_clone.player_boards[i]["score"] for i in range(game_state.num_players) if i != player_idx
        ]
        average_opponent_score = sum(opponent_scores) / len(opponent_scores)

        # Return the score difference (maximising player's advantage)
        return player_score - average_opponent_score

    def hash_game_state(self, game_state, player_idx, depth):
        """
        Generate a hashable representation of the game state for caching.
        """
        factories_hash = tuple(tuple(factory) for factory in game_state.factories)
        center_pool_hash = tuple(game_state.center_pool)
        boards_hash = tuple(
            (tuple(tuple(row) for row in board["wall"]), tuple(tuple(line) for line in board["pattern_lines"]))
            for board in game_state.player_boards
        )
        return (factories_hash, center_pool_hash, boards_hash, player_idx, depth)

    def minmax_worker(self, args):
        """
        Worker function for multiprocessing to evaluate a single move.
        """
        game_state, action, depth, player_idx, alpha, beta = args
        game_state_clone = deepcopy(game_state)
        try:
            factory_idx, tile, pattern_line_idx = action
            game_state_clone.take_action(player_idx, factory_idx, tile, pattern_line_idx)
            next_player = (player_idx + 1) % game_state.num_players
            score, _ = self.minmax_search(game_state_clone, depth - 1, next_player, alpha, beta)
            return score, action
        except ValueError:
            return float('-inf'), None

    def minmax_search(self, game_state, depth, player_idx, alpha, beta):
        """
        Perform a Minimax search with multiprocessing for parallel move evaluation.
        """
        if time.time() - self.start_time >= self.time_limit:
            return self.evaluate_game_state(game_state, player_idx), None

        self.nodes_explored += 1
        self.achieved_depth = max(self.achieved_depth, self.max_depth - depth)

        state_hash = self.hash_game_state(game_state, player_idx, depth)
        if state_hash in self.transposition_table:
            return self.transposition_table[state_hash]

        if depth == 0 or game_state.is_round_over():
            score = self.evaluate_game_state(game_state, player_idx)
            self.transposition_table[state_hash] = (score, None)
            return score, None

        valid_actions = get_valid_actions(game_state, player_idx)
        is_maximising = (player_idx == 0)

        if is_maximising:
            best_score = float('-inf')
            best_move = None
            with mp.Pool() as pool:
                results = pool.map(self.minmax_worker,
                                   [(game_state, action, depth, player_idx, alpha, beta) for action in valid_actions])
            for score, action in results:
                if score > best_score:
                    best_score = score
                    best_move = action
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break
            self.transposition_table[state_hash] = (best_score, best_move)
            return best_score, best_move

        else:
            best_score = float('inf')
            best_move = None
            with mp.Pool() as pool:
                results = pool.map(self.minmax_worker,
                                   [(game_state, action, depth, player_idx, alpha, beta) for action in valid_actions])
            for score, action in results:
                if score < best_score:
                    best_score = score
                    best_move = action
                beta = min(beta, best_score)
                if beta <= alpha:
                    break
            self.transposition_table[state_hash] = (best_score, best_move)
            return best_score, best_move

    def find_optimal_move(self, game_state, player_idx):
        """
        Find the optimal move for the given player using Minimax with multiprocessing.
        """
        self.start_time = time.time()
        self.transposition_table.clear()
        self.nodes_explored = 0
        self.leaf_nodes = 0
        self.achieved_depth = 0
        _, optimal_move = self.minmax_search(
            game_state, self.max_depth, player_idx, float('-inf'), float('inf')
        )
        print(f"Nodes explored: {self.nodes_explored}, Leaf nodes: {self.leaf_nodes}, Achieved depth: {self.achieved_depth}")
        return optimal_move

