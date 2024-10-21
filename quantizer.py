import torch
import torch.nn as nn
import torch.nn.functional as F

class IndexGumbelVectorQuantizerConfig:
    def __init__(self, num_codevector_groups, num_codevectors_per_group, codevector_dim, conv_dim, diversity_loss_weight=0.1):
        self.num_codevector_groups = num_codevector_groups
        self.num_codevectors_per_group = num_codevectors_per_group
        self.codevector_dim = codevector_dim
        self.conv_dim = conv_dim
        self.diversity_loss_weight = diversity_loss_weight

class IndexGumbelVectorQuantizer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_groups = config.num_codevector_groups
        self.num_vars = config.num_codevectors_per_group
        self.codevector_dim = config.codevector_dim

        # Learned embeddings for each index in each group
        self.embeddings = nn.Parameter(
            torch.FloatTensor(self.num_groups, self.num_vars, self.codevector_dim)
        )
        nn.init.uniform_(self.embeddings, -1, 1)

        self.weight_proj = nn.Linear(config.conv_dim[-1], self.num_groups * self.num_vars)

        self.temperature = 2

    @staticmethod
    def _compute_perplexity(probs):
        marginal_probs = probs.mean(dim=0)
        perplexity = torch.exp(-torch.sum(marginal_probs * torch.log(marginal_probs + 1e-7), dim=-1)).sum()
        return perplexity

    def forward(self, hidden_states):
        batch_size, _, hidden_size = hidden_states.shape
        assert _ == 1, "Input should have shape (batch_size, 1, hidden_size)"

        # Project input to logits for each group
        logits = self.weight_proj(hidden_states)  # (batch_size, 1, num_groups * num_vars)
        logits = logits.view(batch_size, self.num_groups, self.num_vars)

        if self.training:
            # Sample class indices using Gumbel-Softmax
            gumbel_dist = F.gumbel_softmax(logits, tau=self.temperature, hard=True, dim=-1)

            # Compute perplexity
            perplexity = self._compute_perplexity(gumbel_dist)

            # Use gumbel_dist to select embeddings
            selected_embeddings = torch.einsum('bgv,gve->bge', gumbel_dist, self.embeddings)
        else:
            # In inference, use argmax to select indices
            indices = torch.argmax(logits, dim=-1)  # (batch_size, num_groups)
            one_hot = F.one_hot(indices, num_classes=self.num_vars).float()

            # Compute perplexity
            perplexity = self._compute_perplexity(one_hot)

            # Use one_hot to select embeddings
            selected_embeddings = torch.einsum('bgv,gve->bge', one_hot, self.embeddings)

        # selected_embeddings shape: (batch_size, num_groups, codevector_dim)
        return selected_embeddings, perplexity



if __name__ == "__main__":
    config = IndexGumbelVectorQuantizerConfig(
        num_codevector_groups=5,
        num_codevectors_per_group=512,
        codevector_dim=768,
        conv_dim=(768,)
    )
    quantizer = IndexGumbelVectorQuantizer(config)

    hidden_states = torch.randn(2, 1, 768)
    quantized,perplexity = quantizer(hidden_states)
    print(quantized.shape)