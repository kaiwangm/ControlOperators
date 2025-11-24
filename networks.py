import torch
import torch.nn as nn

"""
A few notes and observations on the network structure:

    1. We found that a ResNet style network (provided here as `SkipMLP`) generally works just 
       as well as the concatenative skip connections presented in the paper (and also used in 
       [Shi et al. 2024]). In general it seems the most important aspect is that there is some 
       mechanism for propagation of the input to deeper layers.
       
    2. As noted in the paper, it is generally okay to remove the LayerNorm, however
       we did find that if you make the network very deep then removing the LayerNorm 
       can introduce instability as the untrained outputs can get very large (presumably
       due to the issues noted in this paper: https://arxiv.org/pdf/1901.09321 that apply to 
       both ResNet style networks and concatenative skip connection networks).
"""

class MLP(nn.Module):
    def __init__(self, inp: int, out: int, hidden: int = 1024, depth: int = 10):
        super().__init__()
        layers = []
        d = inp
        for _ in range(depth - 1):
            layers += [nn.Linear(d, hidden), nn.ELU()]
            d = hidden
        layers += [nn.Linear(d, out)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SkipMLP(nn.Module):
    def __init__(self, inp, out, hidden=1024, depth=10):
        super().__init__()
        layers = []
        d = inp
        for i in range(depth - 1):
            layers.append(nn.Linear(d, hidden))
            layers.append(nn.ELU())
            d = hidden
        self.layers = nn.ModuleList(layers)
        self.out = nn.Linear(d, out)

    def forward(self, x):
        h = x
        for i in range(0, len(self.layers), 2):
            h_in = h
            h = self.layers[i](h)
            h = self.layers[i+1](h)
            if h.shape == h_in.shape:
                h = h + h_in
        return self.out(h)
    
class SkipCatMLP(nn.Module):
    def __init__(self, inp, out, hidden=800, depth=8):
        super().__init__()
        layers = []
        
        layers.append(nn.Linear(inp, hidden))
        layers.append(nn.GELU())

        for i in range(depth - 1):
            layers.append(nn.LayerNorm(hidden))
            layers.append(nn.Linear(hidden + inp, hidden))
            layers.append(nn.GELU())

        self.layers = nn.ModuleList(layers)
        self.out = nn.Linear(hidden + inp, out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        layers = self.layers

        h = x
        h = layers[0](h)
        h = layers[1](h)

        for i in range(2, len(layers), 3):
            ln = layers[i]
            lin = layers[i + 1]
            act = layers[i + 2]

            h = ln(h)                      
            h = torch.cat((h, x), dim=-1)
            h = lin(h)
            h = act(h)

        return self.out(torch.cat((h, x), dim=-1))
