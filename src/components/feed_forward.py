from utils.common_imports import nn, math, torch

class PositionwiseFeedforward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedforward, self).__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(dropout)

        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        output = self.linear1(x)
        output = self.relu(output)
        output = self.dropout(output)

        output = self.linear2(output)
        
        return output