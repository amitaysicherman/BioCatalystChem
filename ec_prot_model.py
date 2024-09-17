import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Config
import numpy as np


def get_layers(dims, dropout=0.0):
    layers = torch.nn.Sequential()
    for i in range(len(dims) - 1):
        layers.add_module(f"linear_{i}", torch.nn.Linear(dims[i], dims[i + 1]))
        # layers.add_module(f"bn_{i}", torch.nn.BatchNorm1d(dims[i + 1]))
        if i < len(dims) - 2:
            layers.add_module(f"relu_{i}", torch.nn.ReLU())
        if dropout > 0:
            layers.add_module(f"dropout_{i}", torch.nn.Dropout(dropout))
    return layers

class CustomT5Model(T5ForConditionalGeneration):
    def __init__(self, config: T5Config, lookup_table_file, lookup_len, cutoff_index):
        super().__init__(config)
        lookup_table = np.load(lookup_table_file)
        lookup_dim = lookup_table.shape[1]
        self.lookup_table = nn.Embedding.from_pretrained(torch.tensor(lookup_table), freeze=True).float()
        layers_dims = [lookup_dim] + [config.d_model] * lookup_len
        self.lookup_proj = get_layers(layers_dims, dropout=config.dropout_rate)
        self.cutoff_index = cutoff_index

    def forward(self, input_ids, **kwargs):
        regular_token_mask = input_ids < self.cutoff_index
        lookup_token_mask = input_ids >= self.cutoff_index
        regular_embeddings = self.shared(input_ids.clamp(max=self.cutoff_index - 1))  # Clamp to avoid indexing errors
        lookup_indices = input_ids[lookup_token_mask] - self.cutoff_index
        lookup_embeddings = self.lookup_table(lookup_indices)
        transformed_lookup_embeddings = self.lookup_proj(lookup_embeddings)
        final_embeddings = torch.zeros_like(regular_embeddings).float()
        final_embeddings[regular_token_mask] = regular_embeddings[regular_token_mask]
        final_embeddings[lookup_token_mask] = transformed_lookup_embeddings
        return super().forward(inputs_embeds=final_embeddings, **kwargs)



if __name__ == "__main__":
    model = CustomT5Model(T5Config(), "lookup_table.npy", 4, 1000)
    input_ids = torch.randint(0, 1100, (10, 10))
    outputs = model(input_ids)
    print(outputs