import torch.nn as nn
from torch_scatter import scatter_add
import torch_geometric

from transformers import BertModel

from models.graph_layernorm import LayerNorm
from models.gat import GATSeq
from global_variables import *
from utils import *


class GraphPooling(nn.Module):
    def __init__(self, num_node_features, num_out_features):
        super(GraphPooling, self).__init__()
        channels = num_out_features
        self.gate_nn = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Linear(channels, 1)
        )
        self.node_nn = nn.Sequential(
            nn.Linear(num_node_features, channels),
            nn.ReLU(),
            nn.Linear(channels, channels)
        )

        self.reset_parameters()

    def reset_parameters(self):
        torch_geometric.nn.inits.reset(self.gate_nn)
        torch_geometric.nn.inits.reset(self.node_nn)

    def forward(self, x, batch, size=None):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        size = batch[-1].item() + 1 if size is None else size
        x = self.node_nn(x)
        gate = self.gate_nn(x)
        assert gate.dim() == x.dim() and gate.size(0) == x.size(0)

        gate = torch_geometric.utils.softmax(gate, batch, num_nodes=size)
        out = scatter_add(gate * x, batch, dim=0, dim_size=size)

        return out


class GraphEncoder(nn.Module):
    def __init__(self):
        super(GraphEncoder, self).__init__()

        self.graph_emb_dim = config.hidden_size
        if ("bert" in config.plm_name):
            self.plm = BertModel(config)
        else:
            raise NotImplementedError("Unspported language model: " + config.plm_name)
        for param in self.plm.parameters():
            param.requires_grad = False

        self.graph_layer_norm = LayerNorm(self.graph_emb_dim)

        self.gat_seq = GATSeq(
            in_channels=self.graph_emb_dim,
            out_channels=self.graph_emb_dim,
            edge_attr_dim=self.graph_emb_dim,
            num_ins=config.num_graph_convs,
            gat_heads=4,
            dropout=0.1, gat_negative_slope=0.2, gat_bias=True
        )

        self.pooling = GraphPooling(self.graph_emb_dim, self.graph_emb_dim)

        all_params = sum(param.numel() for param in self.parameters())
        plm_params = sum(param.numel() for param in self.plm.parameters())
        trainable_params = sum(param.numel() for param in self.parameters() if param.requires_grad)
        assert all_params == plm_params + trainable_params, \
            f"{all_params} != {plm_params} + {trainable_params}"
        logger.debug(f"Graph encoder has {trainable_params} / {all_params} trainable parameters")
    
    def forward(self, graph: torch_geometric.data.Data):
        x_encoded = self.plm(
            input_ids=graph.x,
            attention_mask=graph.x_attention_mask
        )['pooler_output']
        x_encoded = self.graph_layer_norm(x_encoded, graph.batch)
        edge_attr_encoded = self.plm(
            input_ids=graph.edge_attr,
            attention_mask=graph.edge_attr_attention_mask
        )['pooler_output']
        x = self.gat_seq(
            x=x_encoded,
            edge_index=graph.edge_index,
            edge_attr=edge_attr_encoded,
        )
        x = self.pooling(x, graph.batch, size=None)
        return x


if __name__ == '__main__':
    from torch_geometric.loader import DataLoader
    from dataset_entry import GraphTextDataset

    model = GraphEncoder()
    model.train()

    dataset = GraphTextDataset('val')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    for batch in dataloader:
        graph, text_enc, label = batch
        graph = graph
        outputs = model(graph)
        break
    
    logger.info("done")
