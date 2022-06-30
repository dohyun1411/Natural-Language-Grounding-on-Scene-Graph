import json
from os.path import join

import numpy as np
from torch.utils.data import Dataset
import torch_geometric

from global_variables import *
from utils import *


def tokenize(
    text,
    max_length,
    add_special_tokens=True
    ):

    return config.tokenizer(
        text,
        add_special_tokens=add_special_tokens,
        padding='max_length',
        max_length=max_length,
        return_attention_mask=True,
        return_tensors='pt'
    )

def preprocess_graph_data(graph: dict):
    assert len(graph.keys()) > 0, "Got empty scene graph!"

    obj_id_list = sorted(graph.keys()) # '0', '1', '2', ...
    obj_id_to_node_idx = {obj_id: node_idx for node_idx, obj_id in enumerate(obj_id_list)}
    obj_mask = np.zeros(len(Label), dtype=np.int32)

    node_words = []
    idx_pair_to_edge_word = {}
    edge_topology_list = []

    for node_idx in range(len(obj_id_list)):
        obj_id = obj_id_list[node_idx]
        obj = graph[obj_id]

        node_word = ' '.join(obj['attributes'][:MAX_NODE_WORD - 1])
        node_word += ' ' + obj['name']
        node_word = ' '.join(node_word.split())
        node_words.append(node_word)

        if config.label_type == 'name':
            label_type = obj['name']
        elif config.label_type == 'color':
            label_type = obj['attributes'][1]

        obj_mask[Label.from_name(label_type)] = 1
    
        for rel in obj['relations']:
            idx_pair = (node_idx, obj_id_to_node_idx[rel['object']])
            if idx_pair in edge_topology_list:
                edge_word = idx_pair_to_edge_word[idx_pair]
                edge_word += ' ' + rel['name']
                idx_pair_to_edge_word[idx_pair] = edge_word
            else:
                edge_word = rel['name']
                idx_pair_to_edge_word[idx_pair] = edge_word
                edge_topology_list.append(idx_pair)

    assert len(idx_pair_to_edge_word) == len(obj_id_list) * (len(obj_id_list) - 1), \
        f"{len(idx_pair_to_edge_word)}, {len(obj_id_list)}"

    edge_words = list(idx_pair_to_edge_word.values())
    # logger.debug(node_words)
    # logger.debug(edge_words)

    # Convert to standard pytorch geometirc format
    node_enc = tokenize(node_words, max_length=GRAPH_WORD_MAX_LEN)
    x = node_enc['input_ids']
    x_attention_mask = node_enc['attention_mask']

    edge_enc = tokenize(edge_words, max_length=GRAPH_WORD_MAX_LEN)
    edge_attr = edge_enc['input_ids']
    edge_attr_attention_mask = edge_enc['attention_mask']
    edge_index = torch.tensor(edge_topology_list, dtype=torch.long)

    obj_mask = torch.from_numpy(obj_mask).unsqueeze(0)

    data = torch_geometric.data.Data(
        x=x,
        edge_index=edge_index.t().contiguous(),
        edge_attr=edge_attr,
        x_attention_mask=x_attention_mask,
        edge_attr_attention_mask=edge_attr_attention_mask,
        obj_mask=obj_mask
    )
    
    return data


# https://stackoverflow.com/questions/7642434/is-there-a-way-to-implement-methods-like-len-or-eq-as-classmethods
class LengthMetaclass(type):
    def __len__(self):
        return self.clslength()

class Label(object, metaclass=LengthMetaclass):
    with open(join(join(DATA_PATH, config.task), LABEL_FILE)) as f:
        label_to_name = f.read().splitlines()
    name_to_label = {name: i for i, name in enumerate(label_to_name)}

    @classmethod
    def clslength(cls):
        return len(Label.label_to_name)

    @staticmethod
    def to_name(label):
        return Label.label_to_name[label]

    @staticmethod
    def from_name(name):
        return Label.name_to_label[name]


class GraphTextDataset(Dataset):
    def __init__(self, config, split='train'):
        self.config = config
        self.split = split

        f = open(join(join(DATA_PATH, config.task), f"my_{split}_scenes.json"))
        self.graphs_json = json.load(f)
        f.close()

        g = open(join(join(DATA_PATH, config.task), f"my_{split}_texts_{config.name}.json"))
        self.texts_json = json.load(g)
        g.close()

        self.input_ids_list = []
        self.attention_mask_list = []
        self.label_list = []
        for texts_json_value in self.texts_json.values():
            text_data_list = texts_json_value['data']
            for text_data in text_data_list:
                text = text_data['text']
                answer_name = text_data['label']
                label = Label.from_name(answer_name)
                self.label_list.append(label)

                tokens = tokenize(text, max_length=TEXT_MAX_LEN)

                self.input_ids_list.append(tokens['input_ids'])
                self.attention_mask_list.append(tokens['attention_mask'])

        self.input_ids_list = torch.cat(self.input_ids_list, dim=0)
        self.attention_mask_list = torch.cat(self.attention_mask_list, dim=0)
        self.label_list = torch.tensor(self.label_list)

        self.n = config.num_name + config.num_attr + config.num_single_rel + \
            config.num_most_rel + config.num_common_sense + config.num_ordinal_rel
        assert len(self.input_ids_list) == len(self.label_list)
        assert len(self.input_ids_list) == self.n * len(self.graphs_json), \
            f"{len(self.input_ids_list)} != {self.n} * {len(self.graphs_json)}"
    
    def __len__(self):
        return len(self.input_ids_list)
    
    def __getitem__(self, index):
        graph_index = index // self.n
        graph = self.graphs_json[str(graph_index)]
        graph = preprocess_graph_data(graph)
        input_ids = self.input_ids_list[index]
        attention_mask = self.attention_mask_list[index]
        text_enc = {'input_ids': input_ids, 'attention_mask': attention_mask}
        label = self.label_list[index]
        return graph, text_enc, label


if __name__ == '__main__':
    from tqdm import tqdm

    logger.debug("Open file")
    with open(join(DATA_PATH, "my_val_scenes.json")) as f:
        graph = json.load(f)
    logger.debug("Opening file done!")

    logger.debug("Convert data")
    data = graph['6']
    new_data = preprocess_graph_data(data)
    logger.debug("Converting done!")

    logger.debug("Test dataset")
    dataset = GraphTextDataset(config, split='val')
    for i in tqdm(range(len(dataset))):
        dataset[i]
    logger.debug("Test done!")

    logger.info("done")
