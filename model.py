from typing import Optional, Dict, Any
from transformers import T5ForConditionalGeneration, T5Config, GenerationConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum


class DaaType(Enum):
    MEAN = 0
    DOCKING = 1
    ATTENTION = 2
    ALL = 3


class DockingAwareAttention(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, daa_type=DaaType.ALL, lin_attn=False):
        super(DockingAwareAttention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.daa_type = daa_type

        # Ensure input dimension is divisible by number of heads
        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"
        self.lin_attn = lin_attn
        if daa_type == DaaType.ATTENTION or daa_type == DaaType.ALL:
            if lin_attn:
                # learn the attention weights directly
                self.lin_w = nn.Linear(input_dim, 1)
            else:
                # Linear layers for Q, K, V
                self.q_proj = nn.Linear(input_dim, input_dim)
                self.k_proj = nn.Linear(input_dim, input_dim)
                self.v_proj = nn.Linear(input_dim, input_dim)

        # Output projection
        self.out_proj = nn.Linear(input_dim, output_dim)

        # Learnable parameters with more stable initialization
        self.alpha = nn.Parameter(torch.tensor(1, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(1, dtype=torch.float32))
        # learn emmbeding for empty lines (1, input_dim)
        self.empty_emb = nn.Embedding(1, output_dim)

    def replace_empty_emb(self, x, docking_scores):
        # X is the representation of (batch_size, 1, input_dim)
        # docking_scores is the docking scores of (batch_size, seq)

        empty_mask = docking_scores.sum(dim=1) == 0  # (batch_size)
        empty_mask = empty_mask.unsqueeze(1)  # (batch_size, 1)
        # Generate the embedding for empty inputs
        empty_emb = self.empty_emb(torch.tensor([0], device=x.device))  # (1, input_dim)
        empty_emb = empty_emb.unsqueeze(0)  # (1, 1, input_dim)
        empty_emb = empty_emb.expand(x.size(0), 1, -1)  # (batch_size, 1, input_dim)
        # Use torch.where for conditional replacement
        x = torch.where(empty_mask.unsqueeze(-1), empty_emb, x)
        return x

    def _forward_mean(self, x, docking_scores, mask=None):
        return x.mean(dim=1).unsqueeze(1)

    def _forward_docking(self, x, docking_scores, mask=None):
        docking_scores = docking_scores.unsqueeze(-1)
        if mask is not None:
            d_mask = mask.bool().unsqueeze(-1)
            docking_scores = docking_scores.masked_fill(~d_mask, 0)
        return (docking_scores * x).sum(dim=1).unsqueeze(1)

    def _forward_attention(self, x, docking_scores, mask=None):
        batch_size, seq_len, _ = x.size()
        if self.lin_attn:
            attn_weights = self.lin_w(x).squeeze(-1)
            if mask is not None:
                attn_mask = mask.bool()
                attn_weights = attn_weights.masked_fill(~attn_mask, float('-inf'))  # (batch_size, seq_len)
            attn_weights = F.softmax(attn_weights, dim=-1)  # (batch_size, seq_len)
            attn_weights = attn_weights.unsqueeze(-1)  # (batch_size, seq_len, 1)
            return (attn_weights * x).sum(dim=1).unsqueeze(1)  # (batch_size, 1, input_dim)

        else:
            Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
            if mask is not None:
                attn_mask = mask.bool().unsqueeze(1).unsqueeze(2)
                attn_weights = attn_weights.masked_fill(~attn_mask, float('-inf'))
            attn_weights = F.softmax(attn_weights, dim=-1)
            context = torch.matmul(attn_weights, V)
            context = context.transpose(1, 2).reshape(batch_size, seq_len, self.input_dim)
            return context[:, 0, :].unsqueeze(1)

    def _forward(self, x, docking_scores, mask=None):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, input_dim)
            docking_scores: Tensor of shape (batch_size, seq_len)
            mask: Optional tensor for masking (batch_size, seq_len)
        Returns:
            Tensor of shape (batch_size, seq_len, output_dim)
        """
        batch_size, seq_len, _ = x.size()
        x_mean = self._forward_mean(x, docking_scores, mask)
        if self.daa_type == DaaType.MEAN:
            return x_mean
        x_docking = self._forward_docking(x, docking_scores, mask)
        if self.daa_type == DaaType.DOCKING:
            return self.alpha * x_mean + x_docking
        x_attention = self._forward_attention(x, docking_scores, mask)
        if self.daa_type == DaaType.ATTENTION:
            return x_attention
        # ALL
        return self.alpha * x_mean + self.beta * x_docking + x_attention

    def forward(self, x, docking_scores, mask=None):
        res = self._forward(x, docking_scores, mask)
        res = self.out_proj(res)
        res = self.replace_empty_emb(res, docking_scores)
        return res


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
    def __init__(self, config: T5Config, daa_type, add_mode, prot_dim=2560, lin_attn=False):

        super(CustomT5Model, self).__init__(config)
        self.daa_type = DaaType(daa_type)
        self.add_mode = add_mode
        self.docking_attention = DockingAwareAttention(prot_dim, config.d_model, config.num_heads, self.daa_type,
                                                       lin_attn=lin_attn)

    def prep_input_embeddings(self, input_ids, attention_mask, emb, emb_mask, docking_scores):
        input_embeddings = self.shared(input_ids)  # Shape: (batch_size, sequence_length, embedding_dim)
        batch_size, seq_length, emb_dim = input_embeddings.shape
        emb = self.docking_attention(emb, docking_scores, mask=emb_mask)[:, 0]  # CLS token
        emb = emb.unsqueeze(1)
        if not self.add_mode:
            new_input_embeddings = torch.cat([emb, input_embeddings], dim=1)
            # Update attention mask
            emb_attention = torch.ones(batch_size, emb.shape[1], device=attention_mask.device)
            attention_mask = torch.cat([emb_attention, attention_mask], dim=1)
        else:
            new_input_embeddings = input_embeddings + emb
        return new_input_embeddings, attention_mask

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


if __name__ == "__main__":
    # Test the model
    from transformers import PreTrainedTokenizerFast
    from preprocessing.build_tokenizer import get_tokenizer_file_path, get_ec_tokens

    tokenizer = PreTrainedTokenizerFast.from_pretrained(get_tokenizer_file_path())
    new_tokens = get_ec_tokens()
    tokenizer.add_tokens(new_tokens)
    config = T5Config(vocab_size=len(tokenizer.get_vocab()), pad_token_id=tokenizer.pad_token_id,
                      eos_token_id=tokenizer.eos_token_id,
                      decoder_start_token_id=tokenizer.pad_token_id)
    for daa_type in [0, 1, 2, 3]:
        for lin_attn in [0, 1]:
            print(daa_type, lin_attn)
            model = CustomT5Model(config, daa_type, add_mode=False, prot_dim=2560, lin_attn=lin_attn)
            # print number of parameters
            n1 = sum(p.numel() for p in model.parameters())
            print(f"Number of parameters:{n1:,}")
            # number of params in the docking_attention submodule
            n2 = sum(p.numel() for p in model.docking_attention.parameters())
            print(f"Number of parameters in docking_attention:{n2:,}")
            # print number of parameters for each layer in docking_attention
            for name, param in model.docking_attention.named_parameters():
                print(name, f'{param.numel():,}')
            print("==" * 20)
#
