class MinMaxAzulEnv:
    def __init__(self, game_state, agents):
        self.game_state = game_state
        if len(agents) < 2 or len(agents) > 4:
            raise ValueError("The number of agents must be between 2 and 4.")
        self.agents = agents

    def play_game(self):
        """
        Simulate the game until it ends.
        """
        while not self.game_state.is_game_over():
            current_player_idx = self.game_state.current_player
            agent = self.agents[current_player_idx]
            action = agent.choose_action(self.game_state)
            if action:
                factory_idx, tile, pattern_line_idx = action
                self.game_state.take_action(current_player_idx, factory_idx, tile, pattern_line_idx)

        # Perform final scoring and declare winner
        scores = [board["score"] for board in self.game_state.player_boards]
        winner = scores.index(max(scores))
        return scores, winner