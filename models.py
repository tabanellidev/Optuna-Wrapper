from torch import nn

class FCVar(nn.Module):

    def __init__(self, input_dim, output_dim, dropout, hidden_layers_dim = []):
        super(FCVar,self).__init__()
        
        self.flatten = nn.Flatten()

        self.build = []
        for l in range(len(hidden_layers_dim)):
            self.build.append(nn.Linear(input_dim, hidden_layers_dim[l]))
            self.build.append(nn.BatchNorm1d(hidden_layers_dim[l]))
            self.build.append(nn.ReLU())
            self.build.append(nn.Dropout(dropout))
            input_dim = hidden_layers_dim[l]

        self.layers = nn.ModuleList(self.build)
        self.last_linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.flatten(x)
        
        for layer in self.layers:
            out = layer(out)

        out = self.last_linear(out)

        return out
