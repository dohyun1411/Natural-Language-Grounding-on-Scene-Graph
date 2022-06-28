import torch.nn as nn
import torch.nn.functional as F
import torch_geometric

from transformers import BertModel

from models.graph_layernorm import LayerNorm
from models.gat import GAT
from models.gnn import GraphEncoder, GraphPooling
from models.lm import BertPrefixForSequenceClassificationWithSkip
from global_variables import *
from utils import *


class MyModel(nn.Module):
    def __init__(self, config):
        super(MyModel, self).__init__()
        self.lm = BertPrefixForSequenceClassificationWithSkip(config)

        self.graph_emb_dim = config.hidden_size
        if ("bert" in config.plm_name):
            self.plm = BertModel(config)
        else:
            raise NotImplementedError("Unspported language model: " + config.plm_name)
        for param in self.plm.parameters():
            param.requires_grad = False
        self.graph_layer_norm = LayerNorm(self.graph_emb_dim)

        self.convs = nn.ModuleList([GAT(
            in_channels=self.graph_emb_dim,
            out_channels=self.graph_emb_dim,
            edge_in_channels=self.graph_emb_dim,
            heads=4,
            concat=False,
            negative_slope=0.2,
            dropout=0.1,
            bias=True
        ) for _ in range(config.num_graph_convs)])
        self.pooling = GraphPooling(self.graph_emb_dim, self.graph_emb_dim)
        self.bns = nn.ModuleList([nn.BatchNorm1d(self.graph_emb_dim) for _ in range(config.num_graph_convs - 1)]) 
        self.dropout = 0.1

        all_params = sum(param.numel() for param in self.parameters())
        trainable_params = sum(param.numel() for param in self.parameters() if param.requires_grad)
        logger.info(f"Model parameters: {trainable_params} / {all_params} = {trainable_params / all_params:.2f}")
    
    def forward(self, graph, input_ids, attention_mask, labels, return_dict=None):
        # TODO: pooler output?
        x_encoded = self.plm(
            input_ids=graph.x,
            attention_mask=graph.x_attention_mask
        )['pooler_output']
        x_encoded = self.graph_layer_norm(x_encoded, graph.batch)
        edge_attr_encoded = self.plm(
            input_ids=graph.edge_attr,
            attention_mask=graph.edge_attr_attention_mask
        )['pooler_output']

        size = graph.batch[-1].item() + 1
        graph_prefix = torch.ones([config.prefix_len, size, self.graph_emb_dim])
        h = x_encoded
        # TODO: Use 0 conv(done). BN, ReLU, Droupout?
        graph_prefix[0] = self.pooling(h, graph.batch, size=None)
        for i in range(config.num_graph_convs):
            h = self.convs[i](
                x=h,
                edge_index=graph.edge_index,
                edge_attr=edge_attr_encoded
            )

            # do BN, ReLU, Droupout in-between all conv layers
            if i != config.num_graph_convs - 1:
                h = self.bns[i](h)
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
            
            graph_prefix[i + 1] = self.pooling(h, graph.batch, size=None)

        output = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            graph_prefix=graph_prefix.to(config.device),
            obj_mask=graph.obj_mask,
            return_dict=return_dict
        )
        
        return output


if __name__ == '__main__':
    from torch_geometric.loader import DataLoader
    from dataset_entry import GraphTextDataset
    
    model = MyModel(config)
    
    logger.info(f"Device: {config.device}")
    logger.info(f"Count of using GPUs: {torch.cuda.device_count()}")
    model = model.to(config.device)
    model.train()

    train_dataset = GraphTextDataset(config, split='train')
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size)

    for data in train_dataloader:
        graph, text, labels = data
        graph = graph.to(config.device)
        input_ids = text['input_ids'].to(config.device)
        attention_mask = text['attention_mask'].to(config.device)
        labels = labels.to(config.device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            graph=graph,
            return_dict=True
        )
        break

    logger.info("done")
