import torch
import torch.nn as nn

class KAN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(KAN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Outer function Φ
        self.phi = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=1)
        )
        
        # Inner functions ψ
        self.psi = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU()
            ) for _ in range(output_dim)
        ])

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        psi_outputs = [psi(x) for psi in self.psi]
        psi_concat = torch.stack(psi_outputs, dim=1)
        output = self.phi(psi_concat.sum(dim=1))
        return output
