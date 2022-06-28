import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss

from transformers import BertModel, BertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

from dataset_entry import Label
from models.prefix_encoder import PrefixEncoder
from global_variables import *
from utils import *


class BertPrefixForSequenceClassificationWithSkip(BertPreTrainedModel):
    def __init__(self, config):
        super(BertPrefixForSequenceClassificationWithSkip, self).__init__(config)
        self.num_labels = len(Label)
        self.config = config
        self.bert = BertModel(config).to(config.device)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        for param in self.bert.parameters():
            param.requires_grad = False
        
        self.prefix_len = config.prefix_len
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads

        self.prefix_encoders = nn.ModuleList([PrefixEncoder(config) for _ in range(self.prefix_len)])

        all_params = sum(param.numel() for param in self.parameters())
        plm_params = sum(param.numel() for param in self.bert.parameters())
        trainable_params = sum(param.numel() for param in self.parameters() if param.requires_grad)
        assert all_params == plm_params + trainable_params, \
            f"{all_params} != {plm_params} + {trainable_params}"
        logger.info(f"Language Model parameters: {trainable_params} / {all_params} = {trainable_params / all_params:.2f}")
    
    def get_prompt(self, graph_prefix, batch_size):

        past_key_values = torch.cat([self.prefix_encoders[i](graph_prefix[i]) for i in range(self.prefix_len)], dim=1)

        past_key_values = past_key_values.view(
            batch_size, # 4
            self.prefix_len, # 4
            self.n_layer * 2, # 12 * 2 
            self.n_head, # 12
            self.n_embd # 64
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4])
        past_key_values = past_key_values.split(2)
        return past_key_values

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        graph_prefix=None,
        obj_mask=None
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = input_ids.shape[0]
        past_key_values = self.get_prompt(graph_prefix, batch_size)
        prefix_attention_mask = torch.ones(batch_size, self.prefix_len).to(self.bert.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=past_key_values,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        logits = logits.masked_fill_(obj_mask.eq(0), value=-1e10) # masking

        loss = None
        if labels is not None:
            if self.config.problem_type == "single-label-classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi-label-classification":
                raise NotImplementedError(f"Not supported problem type: {self.config.problem_type}")
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
            else:
                raise NotImplementedError(f"Not supported problem type: {self.config.problem_type}")

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
