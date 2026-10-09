"""Small causal name model inspired by karpathy/makemore's MLP.

Input: preceding characters only, [batch, context_length] integer IDs.
Output: unnormalized next-character logits, [batch, vocab_size].
ID 0 is left padding/start and the end-of-name prediction target.
"""
import torch
from torch import nn

# EVOLVE-BLOCK-START
class NameMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config["vocab_size"], config["embedding_size"])
        self.layers = nn.Sequential(
            nn.Linear(config["context_length"] * config["embedding_size"], config["hidden_size"]),
            nn.Tanh(),
            nn.Linear(config["hidden_size"], config["vocab_size"]),
        )

    def forward(self, context):
        return self.layers(self.embedding(context).flatten(1))


def build_model(config) -> nn.Module:
    return NameMLP(config)
# EVOLVE-BLOCK-END
