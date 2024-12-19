from game.GameState_class import GameState
from MinMax.MinMaxAgent_class import MinMaxAgent

class MinMaxAzulEnv:
    def __init__(self, num_players, agent_depth):
        self.num_players = num_players
        self.agents = [MinMaxAgent(agent_depth) for _ in range(num_players)]
        self.game_state = GameState()

    def play_game(self):
        self.game_state.reset()
        while not self.game_state.is_game_over():
            print(f"--- Round {self.game_state.round_number} ---")
            player_idx = self.game_state.current_player
            print(f"Player {player_idx + 1}'s turn:")
            optimal_move = self.agents[player_idx].find_optimal_move(self.game_state, player_idx)
            if optimal_move:
                factory_idx, tile, pattern_line_idx = optimal_move
                print(f"  Chose move: Factory {factory_idx}, Tile {tile}, Pattern Line {pattern_line_idx}")
                self.game_state.take_action(player_idx, factory_idx, tile, pattern_line_idx)
            else:
                print("No valid moves available.")
            print(self.game_state)

        print("--- Final Scores ---")
        for player_idx, board in enumerate(self.game_state.player_boards):
            print(f"Player {player_idx + 1}: {board['score']} points")
