import torch.nn as nn

class PredefinedModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        """
        Neural network with predefined architecture.
        :param input_dim: Number of input features.
        :param output_dim: Number of output actions.
        :param hidden_dim: Number of neurons in the hidden layer.
        """
        super(PredefinedModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x

    def copy(self):
        """
        Create a deep copy of the model for mutation.
        """
        new_model = PredefinedModel(self.fc1.in_features, self.fc4.out_features, self.fc1.out_features)
        new_model.load_state_dict(self.state_dict())
        return new_model
