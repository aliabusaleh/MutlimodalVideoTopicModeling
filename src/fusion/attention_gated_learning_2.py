import torch
import torch.nn as nn
"""
https://arturmagalhaes.com/research/python/2025/10/23/attention-mechanisms.html

https://github.com/nestor-sun/mcoattention
"""

"""
    3 Modalities as "triangle" with undirected edges:
        => compute attention in any direction between the modalities simultaneously with coupled parameters

    Co-Attention:
        Modality A uses its own Q, but attends to the other modalities K.
        "Coupled" as: each modalities K and V serves as Context for the other modalities Q.
 """
class AddNorm(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x_old, x_new):
        return self.norm(self.dropout(x_new) + x_old)


def attention_matmul(q,k,v):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)/math.sqrt(d_k))
    #print('QK dimension', scores.shape)
    attention = F.softmax(scores, dim=-1)
    return torch.matmul(attention, v), attention

class MultiHeadCoAttention(nn.Module):

    def __init__(self, num_heads, embedding_dim):
       super(MultiHeadCoattention,self ).__init__()
       assert embedding_dim % num_heads == 0
       self.d_k = int(embedding_dim/num_heads)
       self.num_heads = num_heads
       self.linears = nn.ModuleList(
           [nn.Linear(embedding_dim, embedding_dim)for _ in range(3)]
       )
       self.num_modalities = 3


    def forward(self, query, key, value):
       num_batches = query.size(0)
       query, key, value = [
           linear(x).view(
               num_batches
               -1,
               self.num_heads,
               self.d_k
           ).transpose(1,2) for linear, x in zip(self.linears, (query, key, value))]

       x, attention = attention_matmul(query, key, value)

       x = x.transpose(1,2).contiguous().view(num_batches, self.num_modalities, self.num_heads*self.d_k)
       return x


class CoAttention(nn.Module):
    def __init__(self, heads, dimensions, droput, num_modalities):
        super(CoAttention, self).__init__()
        self.coattention = MultiHeadCoAttention(heads, dimensions, num_modalities)
        self.norm = AddNorm(d, droput)
        self.linear = nn.Linear(d,d)

    def forward(self, inputs):
        out = self.coattention(inputs, inputs, inputs)
        out = self.norm(inputs, out)
        out_linear = self.linear(out)
        out = self.norm(out, out_linear)
        return out


class Multimodal_Coattention(nn.Module):
    def __init__(self, heads, d, modality_num, dropout: float = 0.1):
        super(Multimodal_Coattention, self).__init__()
        self.attentions = nn.ModuleList([SelfAttention(heads, d, dropout) for _ in range(modality_num)])
        self.coattention = Coattention(heads, d, dropout, modality_num)
        self.linear = nn.Linear(modality_num * d, d)

    def forward(self, inputs):
        out = []
        for m_num, attention in enumerate(self.attentions):
            out.append(attention(inputs[:, m_num, :]))

        out = torch.stack(out, dim=1)
        out = self.coattention(out).reshape(out.shape[0], -1)
        out = self.linear(out)
        return out