import torch
import torch.nn as nn


# Controller Architecture ##############################################################################################
class Controller(nn.Module):
    def __init__(self, hidden_dim, input_dim, num_layers, num_classes):
        super(Controller, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x, prev_states):
        H_prev, C_prev = prev_states
        x, (H, C) = self.lstm(x, (H_prev, C_prev))
        x = torch.relu(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.softmax(x, dim=2)
        return x, (H, C)
