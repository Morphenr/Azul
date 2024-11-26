import torch
import torch.optim as optim
import random
from ml.DQN_class import DQN
import torch.nn as nn

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

    def select_action_index(self, state, valid_actions):
        """
        Select an action from the valid actions using an epsilon-greedy policy.
        """

        # Filter out invalid actions (padded actions)
        valid_action_indeces = [idx for idx, action in enumerate(valid_actions) if action != None]

        if not valid_action_indeces:
            raise ValueError("No valid actaions available to select from")
        
        if random.random() < self.epsilon:
            selected_index = random.choice(valid_action_indeces)  # Explore by selecting a random valid action
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.q_network(state_tensor).detach().numpy()
                # Filter Q-values to only valid actions
                valid_q_values = [(q_values[0, action_idx], action_idx) for action_idx in valid_action_indeces]
                selected_index = max(valid_q_values, key=lambda x: x[0])[1]

        return selected_index

    def update(self, state, action_index, reward, next_state, done):
        """
        Update the Q-network using the Bellman equation.
        """
        print(f"State: {state}")

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
