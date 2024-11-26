import torch
import torch.optim as optim
import random
from ml.DQN_class import DQN
import torch.nn as nn
import numpy as np

class AzulAgent:
    
    def __init__(self, input_dim, action_dim, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1):
        self.q_network = DQN(input_dim, action_dim)
        self.target_network = DQN(input_dim, action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def mask_invalid_actions(self, q_values, valid_action_indices):
        """
        Mask invalid actions by setting their Q-values to a very low value.
        """

        mask = np.ones(q_values.shape, dtype=bool)
        mask[valid_action_indices] = False
        q_values[mask] = -np.inf  # Set invalid actions to a very low value
        return q_values

    def select_action_index(self, state, env, player_idx):
        """
        Select an action index using epsilon-greedy policy, with masking for valid actions.
        """
        valid_action_indices = env.get_valid_action_indices()

        if not valid_action_indices:
            raise ValueError("No valid actions available to select from.")

        if random.random() < self.epsilon:
            selected_index = random.choice(valid_action_indices)  # Explore
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.q_network(state_tensor).detach().numpy()[0]
                q_values = self.mask_invalid_actions(q_values, valid_action_indices)
                selected_index = np.argmax(q_values)  # Exploit

        return selected_index


    def update(self, state, action_index, reward, next_state, done):
        """
        Update the Q-network using the Bellman equation.
        """

        if isinstance(action_index, tuple):
            raise ValueError(f"Expected action as an integer index, but got tuple: {action_index}")

        state = torch.FloatTensor(state).unsqueeze(0)
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        reward = torch.FloatTensor([reward])
        done = torch.FloatTensor([done])

        # Compute target Q-value
        with torch.no_grad():
            max_next_q = torch.max(self.target_network(next_state))
            target_q = reward + self.gamma * max_next_q * (1 - done)

        # Compute current Q-value
        current_q = self.q_network(state)[0, action_index]

        # Update Q-network
        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
