import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Config
import numpy as np
from preprocessing.ec_to_vec import EC2Vec
from typing import List, Any
from transformers import PreTrainedTokenizerFast
import re


def get_ec_tokens_ids(tokenizer:PreTrainedTokenizerFast):


def get_layers(dims, dropout=0.0):
    layers = torch.nn.Sequential()
    for i in range(len(dims) - 1):
        layers.add_module(f"linear_{i}", torch.nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.add_module(f"relu_{i}", torch.nn.ReLU())
        if dropout > 0:
            layers.add_module(f"dropout_{i}", torch.nn.Dropout(dropout))
    return layers


class CustomT5Model(T5ForConditionalGeneration):
    def __init__(self, config: T5Config, lookup_len, lookup_indexes: List):
        super().__init__(config)
        self.ec_to_vec = EC2Vec()
        lookup_dim = self.ec_to_vec.prot_dim
        layers_dims = [lookup_dim] + [config.d_model] * lookup_len
        self.lookup_proj = get_layers(layers_dims, dropout=config.dropout_rate)
        self.lookup_indexes = torch.Tensor(lookup_indexes).long()

    def forward(self, input_ids, **kwargs):
        lookup_token_mask = torch.zeros_like(input_ids).bool()
        for idx in self.lookup_indexes:
            lookup_token_mask |= input_ids == idx
        regular_token_mask = ~lookup_token_mask
        regular_embeddings = self.shared(input_ids)
        lookup_embeddings = self.ec_to_vec.ids_to_vecs(input_ids[lookup_token_mask])
        transformed_lookup_embeddings = self.lookup_proj(lookup_embeddings)
        final_embeddings = torch.zeros_like(regular_embeddings).float()
        final_embeddings[regular_token_mask] = regular_embeddings[regular_token_mask]
        final_embeddings[lookup_token_mask] = transformed_lookup_embeddings
        return super().forward(inputs_embeds=final_embeddings, **kwargs)
