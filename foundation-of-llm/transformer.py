import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, d_model: int, d_k: int = None):
        super(Attention, self).__init__()
        if d_k is None:
            d_k = d_model

        self.d_k = d_k

        self.w_q = nn.Linear(d_model, d_k, bias=False)
        self.w_k = nn.Linear(d_model, d_k, bias=False)
        self.w_v = nn.Linear(d_model, d_k, bias=False)

    def forward(self, x: torch.Tensor):
        """
        Attention(x) = softmax(Q @ K^T / sqrt(d_k)) @ V

        Args:
            x : [batch_size, seq_len, emb_size]
        """
        Q = self.w_q(x)  # [batch_size, seq_len, d_k]
        K = self.w_k(x)
        V = self.w_v(x)

        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / (self.d_k**0.5)

        attention_weight = torch.softmax(
            scores, dim=-1
        )  # make sure sum(weight, dim=-1) is all 1.0

        output = torch.matmul(attention_weight, V)

        return output, attention_weight

