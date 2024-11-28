import numpy as np

class NeatAgent:
    def __init__(self, net):
        self.net = net

    def select_action_index(self, state, env):
        # Convert state to a NumPy array if it's not already
        input_state = np.array(state, dtype=np.float32)

        # Activate the network to get output values
        output = self.net.activate(input_state)

        # Get valid action indices
        valid_action_indices = env.get_valid_action_indices()
        if not valid_action_indices:
            return None  # No valid actions

        # Mask invalid actions
        output_array = np.array(output)
        masked_output = np.full_like(output_array, -np.inf)
        masked_output[valid_action_indices] = output_array[valid_action_indices]

        # Select the action with the highest value
        action_index = int(np.argmax(masked_output))
        return action_index
