class NeatAgent:
    def __init__(self, net, action_space_manager, game_state_encoder):
        """
        Initialize the agent with a NEAT neural network and an ActionSpaceManager.

        :param net: The neural network created from the winning genome.
        :param action_space_manager: An instance of ActionSpaceManager for action translation.
        """
        self.net = net
        self.action_space_manager = action_space_manager
        self.game_state_encoder = game_state_encoder

    def retrieve_activations(self, encoded_board_state):
        """
        Use the neural network to compute activation values.

        :param encoded_board_state: Encoded features of the board state.
        :return: The output of the neural network (action preferences).
        """
        return self.net.activate(encoded_board_state)

    def select_action(self, game_state):
        """
        Use the neural network to select the best action for the current state.

        :param encoded_board_state: Encoded features of the board state.
        :param game_state: The current game state object.
        :return: The most preferred valid action.
        """
        # Retrieve activations from the neural network
        encoded_board_state = self.game_state_encoder.encode(game_state)
        activations = self.retrieve_activations(encoded_board_state)

        # Use ActionSpaceManager to get sorted valid actions
        sorted_actions = self.action_space_manager.get_sorted_actions(activations, game_state)

        # Return the most preferred valid action
        if sorted_actions:
            return sorted_actions[0]  # Return the top action
        else:
            raise ValueError("No valid actions available for the given game state.")
