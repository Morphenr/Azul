import math
import time
import random
from copy import deepcopy
from collections import defaultdict
from helper_functions.helper_functions import get_valid_actions

class AzulAgent:
    def __init__(self, player_idx: int, time_limit: float = 3.0, monte_carlo_samples: int = 10):
        """
        Initialises the Azul agent.

        Args:
            player_idx (int): The index of the player this agent represents.
            time_limit (float): Time limit in seconds for decision-making.
            monte_carlo_samples (int): Number of simulations for stochastic evaluations.
        """
        self.player_idx = player_idx
        self.time_limit = time_limit
        self.monte_carlo_samples = monte_carlo_samples
        self.start_time: float = 0.0

        # Track nodes visited (all nodes) and leaves visited at each depth
        self.node_count_by_depth = defaultdict(int)
        self.leaf_count_by_depth = defaultdict(int)

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

        # Keep deepening until we run out of time
        while (time.time() - self.start_time) < self.time_limit:
            # Clear counts for this iteration
            self.node_count_by_depth.clear()
            self.leaf_count_by_depth.clear()

            try:
                value, action = self.minimax(
                    game_state=game_state,
                    depth=depth,
                    alpha=-math.inf,
                    beta=math.inf,
                    maximizing_player=True
                )

                # Print out both the leaf and node counts for each sub-depth
                print(f"Leaf and node counts after depth {depth}:")
                for d in range(depth + 1):
                    node_count = self.node_count_by_depth.get(d, 0)
                    leaf_count = self.leaf_count_by_depth.get(d, 0)
                    print(f"  Depth {d}: {leaf_count} leaves / {node_count} nodes")

                if value > best_value:
                    best_value = value
                    best_action = action

                depth += 1

            except Exception as e:
                print(f"Depth {depth} aborted: {e}")
                break

        print(
            f"Max Depth Reached: {depth - 1}, "
            f"Best Value: {best_value}, "
            f"Best Action: {best_action}"
        )
        return best_action

    def minimax(self, game_state, depth: int, alpha: float, beta: float, maximizing_player: bool) -> tuple:
        """
        Performs the minimax algorithm with alpha-beta pruning, using move ordering.

        Args:
            game_state: The current game state.
            depth (int): Current depth in the minimax tree.
            alpha (float): Alpha value for pruning.
            beta (float): Beta value for pruning.
            maximizing_player (bool): Whether the current layer is maximising or minimising.

        Returns:
            tuple: (evaluation_score, best_action).
        """
        # Whenever we visit a node at 'depth', count it
        self.node_count_by_depth[depth] += 1

        # Check if we've exceeded the time limit
        if (time.time() - self.start_time) >= self.time_limit:
            # This node is effectively a leaf because time ran out
            self.leaf_count_by_depth[depth] += 1
            return self.evaluate(game_state, self.player_idx), None

        # Base case: depth is zero or game is over
        if depth == 0 or game_state.is_game_over():
            self.leaf_count_by_depth[depth] += 1
            return self.evaluate(game_state, self.player_idx), None

        current_player = game_state.current_player
        valid_actions = get_valid_actions(game_state, current_player)

        # If no valid actions, it's also effectively a leaf
        if not valid_actions:
            self.leaf_count_by_depth[depth] += 1
            return self.evaluate(game_state, self.player_idx), None

        # --- Move Ordering step ---
        # Order the moves so that more promising ones are evaluated first in alpha–beta pruning.
        ordered_actions = self.order_moves(game_state, valid_actions, current_player, maximizing_player)

        if maximizing_player:
            max_eval = -math.inf
            best_action = None

            for action in ordered_actions:
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

            for action in ordered_actions:
                new_state = self.simulate_action(game_state, action, current_player)
                eval_value, _ = self.minimax(new_state, depth - 1, alpha, beta, True)
                if eval_value < min_eval:
                    min_eval = eval_value

                beta = min(beta, eval_value)
                if beta <= alpha:
                    break

            # Minimising player doesn't choose the action for our agent
            return min_eval, None

    def order_moves(self, game_state, valid_actions, current_player, maximizing_player):
        """
        Orders valid actions by a quick, shallow evaluation score. For the maximising player,
        moves with higher heuristic scores come first; for the minimising player, lower scores
        come first. This helps alpha–beta pruning.

        Args:
            game_state: The current game state.
            valid_actions (list): List of valid actions.
            current_player (int): The player about to move (could be our agent or an opponent).
            maximizing_player (bool): Whether we are in a maximising or minimising layer.

        Returns:
            list: The valid_actions list, sorted by a shallow heuristic evaluation.
        """
        # We can define a lightweight heuristic that examines the immediate
        # score (or partial board state) resulting from each move.
        # For efficiency, we only do a shallow “simulate & evaluate”.
        scored_actions = []
        for action in valid_actions:
            shallow_state = self.simulate_action(game_state, action, current_player)
            score = self.evaluate_shallow(shallow_state, self.player_idx)
            scored_actions.append((score, action))

        # Sort by descending order if maximising, ascending if minimising
        reverse_sort = True if maximizing_player else False
        scored_actions.sort(key=lambda x: x[0], reverse=reverse_sort)

        return [action for score, action in scored_actions]

    def evaluate_shallow(self, game_state, player_idx) -> float:
        """
        A quick evaluation that attempts to measure how good the state is for our agent right now,
        without doing a full round or multiple Monte Carlo simulations. This is to help
        move ordering, not to be a full evaluation.

        You can improve this for domain-specific strategies.
        """
        # For a quick approach, just get the current score from the board as a shallow heuristic.
        # Or you could check how many pattern lines are nearly complete, the negative penalty for floor tiles, etc.
        board = game_state.player_boards[player_idx]
        return board.get("score", 0.0)

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

            if future_scores:
                expected_future_score = sum(future_scores) / len(future_scores)
                return current_score + expected_future_score
            return current_score

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

        # Combine remaining tiles in the bag and discard pile
        all_tiles = new_state.bag_tiles + new_state.discard_tiles
        random.shuffle(all_tiles)

        # Distribute tiles into factories
        factory_count = len(new_state.factories)
        factory_size = new_state.factory_size

        new_state.factories = [
            all_tiles[i * factory_size : (i + 1) * factory_size]
            for i in range(factory_count)
        ]

        # Remaining tiles go back to the bag
        remaining_tiles = all_tiles[factory_count * factory_size :]
        new_state.bag_tiles = remaining_tiles

        # Clear discard since we've reshuffled everything
        new_state.discard_tiles = []

        return new_state
