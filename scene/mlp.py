
import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, in_size, out_size, mid_size, mid_layer_num):
        super().__init__()
        self.layers = self.get_mlp_layers(in_size, out_size, mid_size, mid_layer_num)

    @staticmethod
    def get_mlp_layers(in_size, out_size, mid_size, mid_layer_num):
        layers = nn.Sequential(
            nn.Linear(in_size, mid_size),
            nn.ReLU(),
        )
        for i in range(mid_layer_num-1):
            layers.append(nn.Linear(mid_size, mid_size))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(mid_size, out_size))
        return layers

    def forward(self, x):
        x = self.layers(x)
        return x
    
class Embedder(nn.Module):
    def __init__(self, in_size, out_size, sigma=1):
        super().__init__()

        assert out_size % 2 == 0
        B = torch.normal(0, 1, (out_size//2, in_size), dtype=torch.float32) * sigma
        B = B * 2 * torch.pi
        self.B = nn.Parameter(B.requires_grad_(False)) 

    def forward(self, x):
        x = torch.einsum('...j,kj->...k', x, self.B)
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        return x
    