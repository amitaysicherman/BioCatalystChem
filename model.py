from typing import Optional, Dict, Any
from transformers import T5ForConditionalGeneration, T5Config, GenerationConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum


class DaaType(Enum):
    ATTENTION = 0
    DOCKING = 1
    MEAN = 2
    ALL = 3


class DockingAwareAttention(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, daa_type=DaaType.ALL):
        super(DockingAwareAttention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.daa_type = daa_type

        # Ensure input dimension is divisible by number of heads
        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"

        # Linear layers for Q, K, V
        self.q_proj = nn.Linear(input_dim, input_dim)
        self.k_proj = nn.Linear(input_dim, input_dim)
        self.v_proj = nn.Linear(input_dim, input_dim)

        # Output projection
        self.out_proj = nn.Linear(input_dim, output_dim)

        # Learnable parameters with more stable initialization
        self.alpha = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

    def forward(self, x, docking_scores, mask=None):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, input_dim)
            docking_scores: Tensor of shape (batch_size, seq_len)
            mask: Optional tensor for masking (batch_size, seq_len)
        Returns:
            Tensor of shape (batch_size, seq_len, output_dim)
        """
        batch_size, seq_len, _ = x.size()

        # Compute mean representation
        x_mean = x.mean(dim=1).unsqueeze(1)  # (batch_size, 1, input_dim)

        # Handle different DAA types
        if self.daa_type == DaaType.MEAN:
            return x_mean

        # Prepare docking scores
        docking_scores = docking_scores.unsqueeze(-1)  # (batch_size, seq_len, 1)

        # Handle mask if provided
        if mask is not None:
            # Ensure mask is boolean and broadcast correctly
            d_mask = mask.bool().unsqueeze(-1)
            docking_scores = docking_scores.masked_fill(~d_mask, 0)

        # Docking type specific processing
        if self.daa_type == DaaType.DOCKING:
            docking_x = (docking_scores * x).sum(dim=1)  # (batch_size, input_dim)
            docking_x = docking_x.unsqueeze(1)  # (batch_size, 1, input_dim)
            return docking_x * self.alpha + x_mean * (1 - self.alpha)

        # Multi-head attention processing for ALL type
        # Project inputs to Q, K, V
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Compute scaled dot-product attention
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # Apply mask if provided
        if mask is not None:
            attn_mask = mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
            attn_weights = attn_weights.masked_fill(~attn_mask, float('-inf'))

        # Softmax attention weights
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Incorporate docking scores
        if self.daa_type != DaaType.ALL:
            # Properly broadcast docking scores
            docking_scores_expanded = docking_scores.view(batch_size, 1, seq_len, 1).expand(
                -1, self.num_heads, -1, -1
            )
            attn_weights = (1 - self.beta) * attn_weights + self.beta * docking_scores_expanded

        # Compute context
        context = torch.matmul(attn_weights, V)  # (batch_size, num_heads, seq_len, head_dim)
        context = context.transpose(1, 2).reshape(batch_size, seq_len, self.input_dim)

        # Final projection
        output = self.out_proj(context)
        return output


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
    def __init__(self, config: T5Config, lookup_len, daa_type, prot_dim=2560):

        super(CustomT5Model, self).__init__(config)
        self.daa_type = DaaType(daa_type)
        layers_dims = [prot_dim] + [config.d_model] * lookup_len
        self.docking_attention = DockingAwareAttention(prot_dim, config.d_model, config.num_heads, daa_type)

    def prep_input_embeddings(self, input_ids, attention_mask, emb, emb_mask, docking_scores):
        input_embeddings = self.shared(input_ids)  # Shape: (batch_size, sequence_length, embedding_dim)
        batch_size, seq_length, emb_dim = input_embeddings.shape
        emb = self.docking_attention(emb, docking_scores, mask=emb_mask)[:, 0]  # CLS token
        emb = emb.unsqueeze(1)
        new_input_embeddings = torch.cat([emb, input_embeddings], dim=1)

        # Update attention mask
        emb_attention = torch.ones(batch_size, emb.shape[1], device=attention_mask.device)
        new_attention_mask = torch.cat([emb_attention, attention_mask], dim=1)
        return new_input_embeddings, new_attention_mask

    def forward(self, input_ids=None, attention_mask=None, labels=None, inputs_embeds=None, encoder_outputs=None,
                emb=None, emb_mask=None, docking_scores=None, **kwargs):
        if encoder_outputs is not None:
            return super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                                   encoder_outputs=encoder_outputs, **kwargs)

        if inputs_embeds is None:
            inputs_embeds, attention_mask = self.prep_input_embeddings(input_ids, attention_mask, emb, emb_mask,
                                                                       docking_scores)
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
                                                                                   model_kwargs["emb"],
                                                                                   model_kwargs["emb_mask"],
                                                                                   model_kwargs["docking_scores"])
        model_kwargs["inputs_embeds"] = inputs_embeds
        return super()._prepare_encoder_decoder_kwargs_for_generation(
            None, model_kwargs, model_input_name, generation_config
        )
