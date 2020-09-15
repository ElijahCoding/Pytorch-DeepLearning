import torch
import torch.nn as nn
import torch.nn.functional as F

# option 1
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        output = self.linear1(x)
        output = self.relu(output)
        output = self.linear2(output)
        output = self.sigmoid(output)
        return output

# option 2
class NerualNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NerualNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output = torch.relu(self.linear(x))
        output = torch.sigmoid(self.linear2(output))
        return output