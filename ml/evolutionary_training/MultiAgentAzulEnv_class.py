from game.GameState_class import GameState
from helper_functions.helper_functions import encode_board_state, get_valid_actions

class MultiAgentAzulEnv:
    def __init__(self, num_players):
        self.num_players = num_players
        self.agents = [None] * num_players
        self.game_state = GameState()
        self.current_player = 0

    def reset(self):
        self.game_state.reset()
        self.current_player = 0
        return self.get_state()

    def set_agents(self, agents):
        if len(agents) != self.num_players:
            raise ValueError("Number of agents must match the number of players.")
        self.agents = agents

    def get_valid_action_indices(self):
        """
        Convert valid actions from `get_valid_actions` into indices for the fixed action space.
        """
        valid_actions = get_valid_actions(self.game_state, self.current_player)
        valid_action_indices = [self.game_state.get_action_space_mapper().action_to_index(action) for action in valid_actions if action is not None]
        return valid_action_indices

    def get_state(self):
        return encode_board_state(self.game_state, self.current_player)

    def step(self, action):
        factory_idx, tile, pattern_line_idx = action
        player_idx = self.current_player

        try:
            self.game_state.take_action(player_idx, factory_idx, tile, pattern_line_idx)
        except ValueError:
            # Invalid action, apply penalty
            raise ValueError("Invalid action")

        is_done = self.game_state.is_game_over()
        return self.get_state(), is_done, {"player": player_idx}

    def play_game(self, max_turns=1000):
        """
        Play a game until it ends or the maximum number of turns is reached.
        """
        state = self.reset()

        while (not self.game_state.is_game_over()) and (self.game_state.round_number <= max_turns):
            agent = self.agents[self.current_player]

            # Get valid action indices for the current player
            valid_action_indices = self.get_valid_action_indices()

            if not valid_action_indices:
                # Skip turn if no valid actions
                self.current_player = (self.current_player + 1) % self.num_players
                continue

            # Agent selects an action index
            action_index = agent.select_action_index(state, self)

            if action_index is None:
                # No valid action, skip turn
                self.current_player = (self.current_player + 1) % self.num_players
                continue

            # Map the action index to the actual action
            action = self.game_state.get_action_space_mapper().index_to_action(action_index)

            # Apply the action and get the new state
            next_state, _, _ = self.step(action)

            # Prepare for the next turn
            state = next_state
            self.current_player = (self.current_player + 1) % self.num_players

            # Handle round completion
            if self.game_state.is_round_over():
                self.game_state.wall_tiling_phase()

        return self.game_state
