import torch

class EvolutionaryAgent:
    def __init__(self, model, action_space_manager, game_state_encoder):
        """
        Initialise the agent with a neural network model and utility classes.
        """
        self.model = model
        self.action_space_manager = action_space_manager
        self.game_state_encoder = game_state_encoder

    def predict(self, encoded_state):
        """
        Forward pass through the neural network model.
        """
        # Convert the input to a PyTorch tensor and ensure correct shape
        encoded_state_tensor = torch.tensor(encoded_state, dtype=torch.float32)
        return self.model(encoded_state_tensor)

    def select_action(self, game_state):
        """
        Select the best action based on model predictions.
        """
        # Encode the game state
        encoded_state = self.game_state_encoder.encode(game_state)

        # Get activations from the model
        activations = self.predict(encoded_state)

        # Use ActionSpaceManager to sort valid actions
        sorted_actions = self.action_space_manager.get_sorted_actions(activations.detach().numpy(), game_state)

        # Return the most preferred valid action
        if sorted_actions:
            return sorted_actions[0]  # Return the top action
        else:
            raise ValueError("No valid actions available for the given game state.")
