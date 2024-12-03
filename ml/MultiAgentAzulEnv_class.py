from helper_functions.GameStateEncoder_class import GameStateEncoder
from ml.ActionSpaceManager_class import ActionSpaceManager
from game.GameState_class import GameState

class MultiAgentAzulEnv:
    def __init__(self, num_players,  agents=None):
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
        self.action_space_manager = ActionSpaceManager(game_state=self.game_state)
        self.game_state_encoder = GameStateEncoder()

    def reset(self):
        """
        Reset the environment to the initial state.
        :return: Encoded state of the game.
        """
        self.game_state.reset()
        return self.game_state

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
        valid_actions = self.action_space_manager.get_valid_actions(self.game_state)
        valid_action_indices = [
            self.action_space_manager.action_to_index(action) for action in valid_actions if action is not None
        ]
        return valid_action_indices

    def get_encoded_state(self):
        """
        Encode the current state of the game.
        :return: Encoded board state.
        """
        return self.game_state_encoder.encode(self.game_state)

    def step(self, action):
        """
        Apply the given action to the game state.
        :param action: Action to take (tuple of factory_idx, tile, pattern_line_idx).
        :return: Tuple (next_state, reward, is_done, info).
        """
        factory_idx, tile, pattern_line_idx = action
        player_idx = self.game_state.current_player

        try:
            self.game_state.take_action(player_idx, factory_idx, tile, pattern_line_idx)
        except ValueError:
            # Invalid action penalty
            print(f"Error: {ValueError.args[0]}. Invalid action: {action} - Player {player_idx} penalised - Who is the current player? {self.game_state.current_player}")
            return self.game_state, self.game_state.is_game_over()

        # Evaluate the board state and check if the game is done
        is_done = self.game_state.is_game_over()
        return self.game_state, is_done

    def play_game(self, max_turns=100):
        """
        Play a complete game with the current agents.
        :param max_turns: Maximum number of turns to play.
        :return: Final game state.
        """
        state = self.reset()
        turn_count = 0

        while not self.game_state.is_game_over() and turn_count < max_turns:
            agent = self.agents[self.game_state.current_player]

            # Get valid action indices for the current player
            valid_action_indices = self.get_valid_action_indices()

            if not valid_action_indices:
                raise ValueError(f"No valid actions available for player {self.game_state.current_player}.")

            # Agent selects an action index
            action = agent.select_action(self.game_state)

            # Apply the action and get the next state
            next_state, is_done = self.step(action)
            #print(self.game_state.__str__())

            # Update state and prepare for the next turn
            state = next_state



        return self.game_state
