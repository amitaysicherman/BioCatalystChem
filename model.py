from typing import Optional, Dict, Any

import torch
from transformers import T5ForConditionalGeneration, T5Config, GenerationConfig
from torch import nn
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


class EnzymaticT5Model(nn.Module):
    def __init__(self, config, lookup_len, protein_embedding_dim=2560, quantization=False, q_groups=5,
                 q_codevectors=512, q_index=0):
        super().__init__()
        self.t5_model = T5ForConditionalGeneration(config)
        layers_dims = [protein_embedding_dim] + [config.d_model] * lookup_len
        self.protein_proj = get_layers(layers_dims, dropout=config.dropout_rate)
        self.quantizer = None
        if quantization:
            if q_index == 0:
                from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2GumbelVectorQuantizer
                from transformers.models.wav2vec2.configuration_wav2vec2 import Wav2Vec2Config
                self.q_config = Wav2Vec2Config()
                self.q_config.num_codevector_groups = q_groups
                self.q_config.num_codevectors_per_group = q_codevectors
                self.q_config.codevector_dim = config.d_model
                self.q_config.conv_dim = (config.d_model,)
                self.q_config.diversity_loss_weight = 0.1
                self.quantizer = Wav2Vec2GumbelVectorQuantizer(self.q_config)
            else:
                from quantizer import IndexGumbelVectorQuantizer, IndexGumbelVectorQuantizerConfig
                self.q_config = IndexGumbelVectorQuantizerConfig(num_codevector_groups=q_groups,
                                                                 num_codevectors_per_group=q_codevectors,
                                                                 codevector_dim=config.d_model,
                                                                 conv_dim=(config.d_model,))
                self.quantizer = IndexGumbelVectorQuantizer(self.q_config)

    def forward(self, input_ids=None, attention_mask=None, labels=None, inputs_embeds=None, encoder_outputs=None,
                emb=None, **kwargs):
        # Encode SMILES input
        encoder_outputs = self.t5_model.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Encode protein vector
        protein_proj = self.protein_proj(emb)
        if protein_proj.ndim == 2:
            protein_proj = protein_proj.unsqueeze(1)

        if self.quantizer is not None:
            protein_proj, perplexity = self.quantizer(protein_proj)
        # Combine SMILES encoding and protein encoding
        combined_encoded = torch.cat([protein_proj, encoder_outputs.last_hidden_state], dim=1)
        attention_mask = torch.cat(
            [torch.ones(protein_proj.shape[0], protein_proj.shape[1], device=attention_mask.device),
             attention_mask], dim=1)
        output = self.t5_model(encoder_outputs=[combined_encoded], attention_mask=attention_mask, labels=labels)
        return output

    # def _prepare_encoder_decoder_kwargs_for_generation(
    #         self,
    #         inputs_tensor: torch.Tensor,
    #         model_kwargs,
    #         model_input_name: Optional[str],
    #         generation_config: GenerationConfig,
    # ) -> Dict[str, Any]:
    #     if generation_config is None:
    #         generation_config = GenerationConfig.from_model_config(self.config)
    #     inputs_embeds, model_kwargs["attention_mask"] = self.prep_input_embeddings(inputs_tensor,
    #                                                                                model_kwargs["attention_mask"],
    #                                                                                model_kwargs["emb"])
    #     model_kwargs["inputs_embeds"] = inputs_embeds
    #     return super()._prepare_encoder_decoder_kwargs_for_generation(
    #         None, model_kwargs, model_input_name, generation_config
    #     )
    #


class CustomT5Model(T5ForConditionalGeneration):
    def __init__(self, config: T5Config, lookup_len, quantization=False, q_groups=5, q_codevectors=512, q_index=0):

        super(CustomT5Model, self).__init__(config)

        self.ec_to_vec = EC2Vec(load_model=False)
        lookup_dim = self.ec_to_vec.prot_dim
        layers_dims = [lookup_dim] + [config.d_model] * lookup_len
        self.lookup_proj = get_layers(layers_dims, dropout=config.dropout_rate)
        self.quantizer = None
        if quantization:
            if q_index == 0:
                from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2GumbelVectorQuantizer
                from transformers.models.wav2vec2.configuration_wav2vec2 import Wav2Vec2Config
                self.q_config = Wav2Vec2Config()
                self.q_config.num_codevector_groups = q_groups
                self.q_config.num_codevectors_per_group = q_codevectors
                self.q_config.codevector_dim = config.d_model
                self.q_config.conv_dim = (config.d_model,)
                self.q_config.diversity_loss_weight = 0.1
                self.quantizer = Wav2Vec2GumbelVectorQuantizer(self.q_config)
                self.quantizer.temperature = 50
            else:
                from quantizer import IndexGumbelVectorQuantizer, IndexGumbelVectorQuantizerConfig
                self.q_config = IndexGumbelVectorQuantizerConfig(num_codevector_groups=q_groups,
                                                                 num_codevectors_per_group=q_codevectors,
                                                                 codevector_dim=config.d_model,
                                                                 conv_dim=(config.d_model,))
                self.quantizer = IndexGumbelVectorQuantizer(self.q_config)

    def prep_input_embeddings(self, input_ids, attention_mask, emb):
        input_embeddings = self.shared(input_ids)  # Shape: (batch_size, sequence_length, embedding_dim)
        batch_size, seq_length, emb_dim = input_embeddings.shape

        # Project the embedding
        emb_projection = self.lookup_proj(emb)  # Shape: (batch_size, 1, embedding_dim)
        if emb_projection.ndim == 2:
            emb_projection = emb_projection.unsqueeze(1)
        if self.quantizer is not None:
            emb_projection, perplexity = self.quantizer(emb_projection)
            if self.quantizer.temperature > 2 and not (
                    torch.isnan(emb_projection).any() or torch.isinf(emb_projection).any()):
                self.quantizer.temperature = self.quantizer.temperature - 0.005

        new_input_embeddings = torch.cat([emb_projection, input_embeddings], dim=1)

        # Update attention mask
        emb_attention = torch.ones(batch_size, emb_projection.shape[1], device=attention_mask.device)
        new_attention_mask = torch.cat([emb_attention, attention_mask], dim=1)
        return new_input_embeddings, new_attention_mask

    def forward(self, input_ids=None, attention_mask=None, labels=None, inputs_embeds=None, encoder_outputs=None,
                emb=None, **kwargs):
        if encoder_outputs is not None:
            return super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                                   encoder_outputs=encoder_outputs, **kwargs)

        if inputs_embeds is None:
            inputs_embeds, attention_mask = self.prep_input_embeddings(input_ids, attention_mask, emb)
        output = super().forward(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, **kwargs)
        return output

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
