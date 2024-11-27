import random
class RandomAgent:
    def __init__(self):
        pass

    def select_action_index(self, state, env):
        """
        Select a random valid action.
        """
        valid_action_indices = env.get_valid_action_indices()
        if not valid_action_indices:
            return None  # No valid actions
        return random.choice(valid_action_indices)
