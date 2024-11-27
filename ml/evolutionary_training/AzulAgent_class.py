import torch
from ml.DQN_class import DQN
import random
import numpy as np

class AzulAgent:
    def __init__(self, input_dim, action_dim):
        self.policy_network = DQN(input_dim, action_dim)
        self.fitness = 0  # To store the agent's fitness after evaluation

    def select_action_index(self, state, env):
        """
        Select an action index based on the policy network, with masking for valid actions.
        """
        valid_action_indices = env.get_valid_action_indices()

        if not valid_action_indices:
            return None  # No valid actions

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.policy_network(state_tensor).detach().numpy()[0]
            q_values = self.mask_invalid_actions(q_values, valid_action_indices)
            selected_indices = np.argwhere(q_values == np.max(q_values)).flatten()
            selected_index = np.random.choice(selected_indices)  # Break ties randomly

        return selected_index

    def mask_invalid_actions(self, q_values, valid_action_indices):
        """
        Mask invalid actions by setting their Q-values to a very low value.
        """
        mask = np.ones(q_values.shape, dtype=bool)
        mask[valid_action_indices] = False
        q_values[mask] = -np.inf  # Set invalid actions to a very low value
        return q_values

    def get_parameters(self):
        """
        Get the parameters of the agent's policy network.
        """
        return self.policy_network.state_dict()

    def set_parameters(self, state_dict):
        """
        Set the parameters of the agent's policy network.
        """
        self.policy_network.load_state_dict(state_dict)

    def clone(self):
        """
        Create a deep copy of the agent.
        """
        clone_agent = AzulAgent(self.policy_network.input_dim, self.policy_network.output_dim)
        clone_agent.set_parameters(self.get_parameters())
        return clone_agent

    def mutate(self, mutation_rate=0.01):
        """
        Apply mutation to the agent's policy network parameters.
        """
        for param in self.policy_network.parameters():
            if random.random() < mutation_rate:
                # Add Gaussian noise
                noise = torch.randn_like(param) * 0.1
                param.data.add_(noise)

    def crossover(self, other_agent):
        """
        Perform crossover with another agent to produce a child agent.
        """
        child_agent = self.clone()
        for param_child, param_self, param_other in zip(child_agent.policy_network.parameters(),
                                                        self.policy_network.parameters(),
                                                        other_agent.policy_network.parameters()):
            # Randomly choose parameters from self or other_agent
            mask = torch.rand_like(param_self) > 0.5
            param_child.data.copy_(param_self.data * mask + param_other.data * (~mask))
        return child_agent

    def save_model(self, filepath):
        """
        Save the agent's model to a file.
        """
        torch.save(self.policy_network.state_dict(), filepath)
        print(f"Agent saved to {filepath}.")
