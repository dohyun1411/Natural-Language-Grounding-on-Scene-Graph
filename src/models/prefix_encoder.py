import torch
import torch.nn as nn

from utils import *


class PrefixEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.prefix_projection = config.prefix_projection
        self.last_hidden_size = config.num_hidden_layers * 2 * config.hidden_size
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            # self.embedding = nn.Embedding(config.pre_seq_len, config.hidden_size)
            self.trans = nn.Sequential(
                nn.Linear(config.hidden_size, config.prefix_hidden_size),
                nn.Tanh(),
                nn.Linear(config.prefix_hidden_size, self.last_hidden_size)
            )
        else:
            raise NotImplementedError("Plase use prefix projection!")
            # self.embedding = nn.Embedding(config.pre_seq_len, config.num_hidden_layers * 2 * config.hidden_size)

    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            # prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix)
        else:
            pass
            # past_key_values = self.embedding(prefix)
        return past_key_values