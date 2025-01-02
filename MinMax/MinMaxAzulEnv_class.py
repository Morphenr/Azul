class MinMaxAzulEnv:
    def __init__(self, game_state, agents, visualiser=None):
        """
        Initialize the Azul environment.
        :param game_state: The initial game state.
        :param agents: List of agents playing the game.
        :param visualiser: Optional visualiser to render the game state.
        """
        self.game_state = game_state
        if len(agents) < 2 or len(agents) > 4:
            raise ValueError("The number of agents must be between 2 and 4.")
        self.agents = agents
        self.visualiser = visualiser

    def play_game(self):
        """
        Simulate the game until it ends, rendering the state after each action.
        :return: Final scores and the winner's index.
        """
        while not self.game_state.is_game_over():
            current_player_idx = self.game_state.current_player
            agent = self.agents[current_player_idx]

            # Render the current state before the action
            if self.visualiser:
                self.visualiser.render(self.game_state)

            action = agent.choose_action(self.game_state)
            if action:
                factory_idx, tile, pattern_line_idx = action
                self.game_state.take_action(current_player_idx, factory_idx, tile, pattern_line_idx)

        # Perform final scoring and declare winner
        scores = [board["score"] for board in self.game_state.player_boards]
        winner = scores.index(max(scores))

        # Render final game state
        if self.visualiser:
            print("Final state reached. Rendering the results...")
            self.visualiser.render(self.game_state)

        return scores, winner
