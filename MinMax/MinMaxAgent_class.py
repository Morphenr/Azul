import math
import time
import random
from copy import deepcopy
from helper_functions.helper_functions import get_valid_actions


class AzulAgent:
    def __init__(self, player_idx: int, time_limit: float = 15.0, monte_carlo_samples: int = 10):
        """
        Initialises the Azul agent.

        Args:
            player_idx (int): The index of the player this agent represents.
            depth (int): Default depth for minimax.
            time_limit (float): Time limit in seconds for decision-making.
            monte_carlo_samples (int): Number of simulations for stochastic evaluations.
        """
        self.player_idx = player_idx

        self.time_limit = time_limit
        self.monte_carlo_samples = monte_carlo_samples  # Number of random simulations
        self.start_time: float = 0.0  # Will be set in choose_action

    def choose_action(self, game_state) -> tuple:
        """
        Decides the best action for the current game state using iterative deepening.

        Args:
            game_state: The current game state.

        Returns:
            tuple: The best action as determined by the agent.
        """
        self.start_time = time.time()
        return self.iterative_deepening(game_state)

    def iterative_deepening(self, game_state) -> tuple:
        """
        Performs iterative deepening to determine the best action within the time limit.

        Args:
            game_state: The current game state.

        Returns:
            tuple: The best action found within the time limit.
        """
        depth = 1
        best_action = None
        best_value = -math.inf

        while (time.time() - self.start_time) < self.time_limit:
            try:
                value, action = self.minimax(
                    game_state=game_state,
                    depth=depth,
                    alpha=-math.inf,
                    beta=math.inf,
                    maximizing_player=True
                )
                if value > best_value:
                    best_value = value
                    best_action = action
                depth += 1
            except Exception as e:
                print(f"Depth {depth} aborted: {e}")
                break

        print(f"Max Depth Reached: {depth - 1}, Best Value: {best_value}, Best Action: {best_action}")
        return best_action

    def minimax(self, game_state, depth: int, alpha: float, beta: float, maximizing_player: bool) -> tuple:
        """
        Performs the minimax algorithm with alpha-beta pruning.

        Args:
            game_state: The current game state.
            depth (int): Current depth in the minimax tree.
            alpha (float): Alpha value for pruning.
            beta (float): Beta value for pruning.
            maximizing_player (bool): Whether the current layer is maximising or minimising.

        Returns:
            tuple: (evaluation_score, best_action).
        """
        # Check time limit
        if (time.time() - self.start_time) >= self.time_limit:
            return self.evaluate(game_state, self.player_idx), None

        # Base case: depth reached or game over
        if depth == 0 or game_state.is_game_over():
            return self.evaluate(game_state, self.player_idx), None

        current_player = game_state.current_player
        valid_actions = get_valid_actions(game_state, current_player)

        if not valid_actions:
            # No available actions, evaluate the state
            return self.evaluate(game_state, self.player_idx), None

        best_action = None

        if maximizing_player:
            max_eval = -math.inf
            for action in valid_actions:
                new_state = self.simulate_action(game_state, action, current_player)
                eval_value, _ = self.minimax(new_state, depth - 1, alpha, beta, False)

                if eval_value > max_eval:
                    max_eval = eval_value
                    best_action = action

                alpha = max(alpha, eval_value)
                if beta <= alpha:
                    break

            return max_eval, best_action
        else:
            min_eval = math.inf
            for action in valid_actions:
                new_state = self.simulate_action(game_state, action, current_player)
                eval_value, _ = self.minimax(new_state, depth - 1, alpha, beta, True)

                if eval_value < min_eval:
                    min_eval = eval_value

                beta = min(beta, eval_value)
                if beta <= alpha:
                    break

            # For the minimising player, we are not storing a 'best_action'
            # because the agent in question only controls the maximising moves.
            return min_eval, None

    def simulate_action(self, game_state, action: tuple, player_idx: int):
        """
        Simulates the result of an action.

        Args:
            game_state: The current game state.
            action (tuple): The action to simulate.
            player_idx (int): The player taking the action.

        Returns:
            A new game state reflecting the action.
        """
        factory_idx, tile, pattern_line_idx = action
        new_state = deepcopy(game_state)
        new_state.take_action(player_idx, factory_idx, tile, pattern_line_idx)
        return new_state

    def evaluate(self, game_state, player_idx: int) -> float:
        """
        Evaluates the game state for the given player.

        Args:
            game_state: The current game state.
            player_idx (int): Index of the player to evaluate.

        Returns:
            float: Estimated score for the player.
        """
        pseudo_game_state = deepcopy(game_state)
        pseudo_game_state.wall_tiling_phase(is_score_evaluation=True)
        board = pseudo_game_state.player_boards[player_idx]

        current_score = board.get("score", 0.0)

        # If the round is ending, add stochastic simulations to estimate future score
        if not pseudo_game_state.factories and not pseudo_game_state.center_pool:
            future_scores = []
            for _ in range(self.monte_carlo_samples):
                simulated_game_state = self.simulate_next_round(pseudo_game_state)
                simulated_game_state.wall_tiling_phase(is_score_evaluation=True)
                future_scores.append(simulated_game_state.player_boards[player_idx].get("score", 0))

            expected_future_score = sum(future_scores) / len(future_scores) if future_scores else 0
            return current_score + expected_future_score

        return current_score

    def simulate_next_round(self, game_state):
        """
        Simulates the randomisation of the next round by replenishing factories and the centre pool.

        Args:
            game_state: The current game state.

        Returns:
            A new game state with factories and the centre pool populated.
        """
        new_state = deepcopy(game_state)

        # Combine remaining tiles in bag and discard pile
        all_tiles = new_state.bag_tiles + new_state.discard_tiles
        random.shuffle(all_tiles)

        # Distribute tiles into factories
        factory_count = len(new_state.factories)
        factory_size = new_state.factory_size

        new_state.factories = [
            all_tiles[i * factory_size: (i + 1) * factory_size]
            for i in range(factory_count)
        ]

        # Remaining tiles go back to the bag
        remaining_tiles = all_tiles[factory_count * factory_size:]
        new_state.bag_tiles = remaining_tiles
        # Clear discard since we've just reshuffled everything
        new_state.discard_tiles = []

        return new_state
