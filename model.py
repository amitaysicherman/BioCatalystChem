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
    def __init__(self, config: T5Config, lookup_len):

        super(CustomT5Model, self).__init__(config)

        self.ec_to_vec = EC2Vec(load_model=False)
        lookup_dim = self.ec_to_vec.prot_dim
        layers_dims = [lookup_dim] + [config.d_model] * lookup_len
        self.lookup_proj = get_layers(layers_dims, dropout=config.dropout_rate)

    def prep_input_embeddings(self, input_ids, attention_mask, emb):
        input_embeddings = self.shared(input_ids)  # Shape: (batch_size, sequence_length, embedding_dim)
        batch_size, seq_length, emb_dim = input_embeddings.shape

        # Project the embedding
        emb_projection = self.lookup_proj(emb)  # Shape: (batch_size, 1, embedding_dim)
        if emb_projection.ndim == 2:
            emb_projection = emb_projection.unsqueeze(1)

        # Concatenate the projected embedding with the input embeddings
        new_input_embeddings = torch.cat([emb_projection, input_embeddings], dim=1)

        # Update attention mask
        emb_attention = torch.ones(batch_size, 1, device=attention_mask.device)
        new_attention_mask = torch.cat([emb_attention, attention_mask], dim=1)

        return new_input_embeddings, new_attention_mask

    def forward(self, input_ids=None, attention_mask=None, labels=None, inputs_embeds=None, encoder_outputs=None,
                emb=None, **kwargs):
        if encoder_outputs is not None:
            return super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                                   encoder_outputs=encoder_outputs, **kwargs)
        if inputs_embeds is None:
            inputs_embeds, attention_mask = self.prep_input_embeddings(input_ids, attention_mask, emb)

        return super().forward(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, **kwargs)

    def _prepare_encoder_decoder_kwargs_for_generation(
            self,
            inputs_tensor: torch.Tensor,
            model_kwargs,
            model_input_name: Optional[str],
            generation_config: GenerationConfig,
    ) -> Dict[str, Any]:
        if generation_config is None:
            generation_config = GenerationConfig.from_model_config(self.config)
        inputs_embeds, model_kwargs["attention_mask"] = self.prep_input_embeddings(inputs_tensor,
                                                                                   model_kwargs["attention_mask"],
                                                                                   model_kwargs["emb"])
        model_kwargs["inputs_embeds"] = inputs_embeds
        return super()._prepare_encoder_decoder_kwargs_for_generation(
            None, model_kwargs, model_input_name, generation_config
        )
