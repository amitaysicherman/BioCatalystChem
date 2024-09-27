from typing import Optional, Dict, Any

import torch
from transformers import T5ForConditionalGeneration, T5Config, GenerationConfig
from preprocessing.ec_to_vec import EC2Vec


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
    def __init__(self, config: T5Config, lookup_len, cutoff_index, ec_tokens_order):
        super(CustomT5Model, self).__init__(config)

        self.ec_to_vec = EC2Vec(load_model=False)
        self.lookup_embeddings = self.ec_to_vec.get_vecs_numpy(ec_tokens_order)
        self.lookup_embeddings = torch.nn.Embedding.from_pretrained(torch.tensor(self.lookup_embeddings), freeze=True)
        self.lookup_embeddings = self.lookup_embeddings.to(torch.float32)
        lookup_dim = self.ec_to_vec.prot_dim
        layers_dims = [lookup_dim] + [config.d_model] * lookup_len
        self.lookup_proj = get_layers(layers_dims, dropout=config.dropout_rate)
        self.cutoff_index = cutoff_index
        self.encoder

    def prep_input_embeddings(self, input_ids):
        regular_token_mask = input_ids < self.cutoff_index
        lookup_token_mask = input_ids >= self.cutoff_index
        regular_embeddings = self.shared(input_ids.clamp(max=self.cutoff_index - 1))  # Clamp to avoid indexing errors
        lookup_indices = input_ids[lookup_token_mask] - self.cutoff_index
        lookup_embeddings = self.lookup_embeddings(lookup_indices)
        transformed_lookup_embeddings = self.lookup_proj(lookup_embeddings)

        final_embeddings = torch.zeros_like(regular_embeddings).float()
        final_embeddings[regular_token_mask] = regular_embeddings[regular_token_mask]
        final_embeddings[lookup_token_mask] = transformed_lookup_embeddings
        return final_embeddings

    def forward(self, input_ids=None, attention_mask=None, labels=None, inputs_embeds=None, **kwargs):
        if inputs_embeds is None:
            inputs_embeds = self.prep_input_embeddings(input_ids)
        return super().forward(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, **kwargs)

    def _prepare_encoder_decoder_kwargs_for_generation(
        self,
        inputs_tensor: torch.Tensor,
        model_kwargs,
        model_input_name: Optional[str],
        generation_config: GenerationConfig,
    ) -> Dict[str, Any]:
        input_embeddings = self.prep_input_embeddings(inputs_tensor)
        return super()._prepare_encoder_decoder_kwargs_for_generation(inputs_tensor, model_kwargs, model_input_name, generation_config, inputs_embeds=input_embeddings)
