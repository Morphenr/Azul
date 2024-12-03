class MultiAgentAzulEnv:
    def __init__(self, num_players, action_space_manager, board_encoder, agents=None):
        """
        Initialize the MultiAgentAzulEnv environment.
        :param num_players: Number of players in the game.
        :param action_space_manager: Instance of the ActionSpaceManager class.
        :param board_encoder: Instance of the GameStateEncoder class.
        :param agents: List of agents for each player (optional).
        """
        self.num_players = num_players
        self.agents = agents if agents else [None] * num_players
        self.game_state = GameState(num_players)
        self.current_player = 0
        self.action_space_manager = action_space_manager
        self.board_encoder = board_encoder

    def reset(self):
        """
        Reset the environment to the initial state.
        :return: Encoded state of the game.
        """
        self.game_state.reset()
        self.current_player = 0
        return self.get_state()

    def set_agents(self, agents):
        """
        Assign agents to players.
        :param agents: List of agents, one for each player.
        """
        if len(agents) != self.num_players:
            raise ValueError("Number of agents must match the number of players.")
        self.agents = agents

    def get_valid_action_indices(self):
        """
        Get valid actions as indices in the fixed action space.
        :return: List of valid action indices.
        """
        valid_actions = self.action_space_manager.get_valid_actions(self.game_state, self.current_player)
        valid_action_indices = [
            self.action_space_manager.action_to_index(action) for action in valid_actions if action is not None
        ]
        return valid_action_indices

    def get_state(self):
        """
        Encode the current state of the game.
        :return: Encoded board state.
        """
        return self.board_encoder.encode(self.game_state)

    def step(self, action):
        """
        Apply the given action to the game state.
        :param action: Action to take (tuple of factory_idx, tile, pattern_line_idx).
        :return: Tuple (next_state, reward, is_done, info).
        """
        factory_idx, tile, pattern_line_idx = action
        player_idx = self.current_player

        try:
            self.game_state.take_action(player_idx, factory_idx, tile, pattern_line_idx)
        except ValueError:
            # Invalid action penalty
            return self.get_state(), -10, False, {"player": player_idx}

        # Evaluate the board state and check if the game is done
        reward = evaluate_board_state(self.game_state, player_idx)
        is_done = self.game_state.is_game_over()
        return self.get_state(), reward, is_done, {"player": player_idx}

    def play_game(self, max_turns=100):
        """
        Play a complete game with the current agents.
        :param max_turns: Maximum number of turns to play.
        :return: Final game state.
        """
        state = self.reset()
        turn_count = 0

        while not self.game_state.is_game_over() and turn_count < max_turns:
            agent = self.agents[self.current_player]

            # Get valid action indices for the current player
            valid_action_indices = self.get_valid_action_indices()

            if not valid_action_indices:
                raise ValueError(f"No valid actions available for player {self.current_player}.")

            # Agent selects an action index
            action_index = agent.select_action_index(state, self, self.current_player)

            # Map the action index to the actual action
            action = self.action_space_manager.index_to_action(action_index)

            # Apply the action and get the next state
            next_state, reward, is_done, _ = self.step(action)

            # Update the agent's strategy
            agent.update(state, action_index, reward, next_state, is_done)

            # Update state and prepare for the next turn
            state = next_state
            turn_count += 1
            self.current_player = (self.current_player + 1) % self.num_players

            # Handle round completion
            if self.game_state.is_round_over():
                self.game_state.wall_tiling_phase()

        return self.game_state
