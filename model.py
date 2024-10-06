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
        # self.lookup_embeddings = self.ec_to_vec.get_vecs_numpy(ec_tokens_order)
        # self.lookup_embeddings = torch.nn.Embedding.from_pretrained(torch.tensor(self.lookup_embeddings), freeze=True)
        # self.lookup_embeddings = self.lookup_embeddings.to(torch.float32)
        lookup_dim = self.ec_to_vec.prot_dim
        layers_dims = [lookup_dim] + [config.d_model] * lookup_len
        self.lookup_proj = get_layers(layers_dims, dropout=config.dropout_rate)
        # self.cutoff_index = cutoff_index

    def prep_input_embeddings(self, input_ids, attention_mask, emb):
        input_embeddings = self.shared(input_ids)  # Shape: (batch_size, sequence_length, embedding_dim)

        batch_size, seq_length, emb_dim = input_embeddings.shape
        seq_length += 1

        # Find the length of each sequence (number of non-padding tokens)
        seq_lengths = attention_mask.sum(dim=1).tolist()  # List of lengths for each sequence in the batch

        new_embeddings = []
        for i, seq_len in enumerate(seq_lengths):
            if (emb[i] == 0).all():
                combined_embeddings = input_embeddings[i]
            else:
                current_embeddings = input_embeddings[i, :seq_len - 1]  # Shape: (seq_len-1, embedding_dim)
                combined_embeddings = torch.cat([current_embeddings, self.lookup_proj(emb[i].unsqueeze(0))], dim=0)
                eos_embedding = input_embeddings[i, seq_len - 1].unsqueeze(0)  # Shape: (1, embedding_dim)
                combined_embeddings = torch.cat([combined_embeddings, eos_embedding], dim=0)
            padding_length = seq_length - combined_embeddings.size(0)
            if padding_length > 0:
                padding = torch.ones(padding_length, emb_dim, device=input_embeddings.device)*self.config.pad_token_id
                combined_embeddings = torch.cat([combined_embeddings, padding], dim=0)
            new_embeddings.append(combined_embeddings)
        # add last attention mask to new embeddings
        new_attention_mask = torch.ones(batch_size, seq_length, device=input_embeddings.device)
        for i, seq_len in enumerate(seq_lengths):
            new_attention_mask[i, seq_len:] = 0

        # Stack the new embeddings to form a batch
        new_input_embeddings = torch.stack(new_embeddings)
        return new_input_embeddings,new_attention_mask  # Shape: (batch_size, sequence_length, embedding_dim)
        #
        # regular_token_mask = input_ids < self.cutoff_index
        # lookup_token_mask = input_ids >= self.cutoff_index
        # regular_embeddings = self.shared(input_ids.clamp(max=self.cutoff_index - 1))  # Clamp to avoid indexing errors
        # lookup_indices = input_ids[lookup_token_mask] - self.cutoff_index
        # lookup_embeddings = self.lookup_embeddings(lookup_indices)
        # transformed_lookup_embeddings = self.lookup_proj(lookup_embeddings)
        #
        # final_embeddings = torch.zeros_like(regular_embeddings).float()
        # final_embeddings[regular_token_mask] = regular_embeddings[regular_token_mask]
        # final_embeddings[lookup_token_mask] = transformed_lookup_embeddings
        # return final_embeddings

    def forward(self, input_ids=None, attention_mask=None, labels=None, inputs_embeds=None, encoder_outputs=None,
                emb=None, **kwargs):
        if encoder_outputs is not None:
            return super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                                   encoder_outputs=encoder_outputs, **kwargs)
        if inputs_embeds is None:
            inputs_embeds,attention_mask = self.prep_input_embeddings(input_ids, attention_mask, emb)

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
        inputs_embeds,model_kwargs["attention_mask"] = self.prep_input_embeddings(inputs_tensor, model_kwargs["attention_mask"], model_kwargs["emb"])
        model_kwargs["inputs_embeds"] = inputs_embeds
        return super()._prepare_encoder_decoder_kwargs_for_generation(
            None, model_kwargs, model_input_name, generation_config
        )
