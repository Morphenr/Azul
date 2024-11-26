from game.GameState_class import GameState
from helper_functions.helper_functions import encode_board_state, simulate_action, calculate_scores, is_game_over, get_valid_actions

class MultiAgentAzulEnv:
    def __init__(self, num_players=2):
        self.num_players = num_players
        self.agents = [None] * num_players
        self.game_state = GameState(num_players)
        self.current_player = 0

    def reset(self):
        self.game_state.reset()
        self.current_player = 0
        return self.get_state()

    def set_agents(self, agents):
        if len(agents) != self.num_players:
            raise ValueError("Number of agents must match the number of players.")
        self.agents = agents

    def get_state(self):
        return encode_board_state(self.game_state)

    def step(self, action):
        factory_idx, tile, pattern_line_idx = action
        player_idx = self.current_player

        try:
            simulate_action(self.game_state, player_idx, factory_idx, tile, pattern_line_idx)
        except ValueError:
            return self.get_state(), -10, False, {"player": player_idx}

        # Access player_boards using dot notation
        reward = calculate_scores(self.game_state.player_boards[player_idx])  # Updated line
        is_done = is_game_over(self.game_state)
        self.current_player = (self.current_player + 1) % self.num_players
        return self.get_state(), reward, is_done, {"player": player_idx}


    def play_game(self, max_turns=100):
        state = self.reset()
        turn_count = 0

        while not self.game_state.is_round_over() and turn_count < max_turns:
            agent = self.agents[self.current_player]
            valid_actions = get_valid_actions(self.game_state, self.current_player)

            if not valid_actions:
                raise ValueError(f"No valid actions available for player {self.current_player}.")

            action_index = agent.select_action_index(state, valid_actions)
            next_state, reward, _, _ = self.step(valid_actions[action_index])
            agent.update(state, action_index, reward, next_state, False)
            state = next_state
            turn_count += 1
            self.current_player = (self.current_player + 1) % self.num_players

        # Perform wall-tiling phase and reset for next round
        self.game_state.wall_tiling_phase()

        # Check if the game is over
        return is_game_over(self.game_state)

