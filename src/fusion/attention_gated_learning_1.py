import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Approach inspired by a combination from:
https://arturmagalhaes.com/research/python/2025/10/23/attention-mechanisms.html
https://github.com/nestor-sun/mcoattention

Adjusted into an approach that actually separates Q and KV instead of stacking them
"""
class AddNorm(nn.Module):
    """
    taken from:
        https://github.com/nestor-sun/mcoattention/transformer_layer.py

    """
    def __init__(self, embed_dims, dropout=0.1):
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dims)

    def forward(self, x_old, x_new):
        return self.norm(self.dropout(x_new) + x_old)


class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dims):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm = nn.LayerNorm(embed_dims)
    def forward(self, x):
        return self.norm(self.ffn(x))



class SelfAttention(nn.Module):
    """
    Inspired from:
       https://github.com/nestor-sun/mcoattention/transformer_layer.py
   """
    def __init__(self, num_heads, embed_dims):
        super(SelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(
            num_heads,
            embed_dims,
            drouput=0.1,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(embed_dims)
        self.linear = nn.Linear(embed_dims, embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)

    def forward(self, i):
        x, attention_weights = self.attention(i, i, i)
        x_norm = self.norm1(i, x)
        x_linear = self.linear(x_norm)
        x_normed = self.norm2(x_norm, x_linear)
        return x_normed, attention_weights


class CoAttention(nn.Module):
    def __init__(self, num_heads, embed_dims):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dims)


    def forward(self, query, key_value):
        attn_output, attn_weights = self.attention(query, key_value, key_value)
        query = self.norm(query + attn_output)
        return query, attn_weights


class ModalityBlock(nn.Module):
    def __init__(self, num_heads, embed_dims, dropout: float = 0.1):
        super().__init__()
        self.self_attentions = SelfAttention(num_heads, embed_dims)
        self.ffns = FeedForwardNetwork(embed_dims)
        self.norms = AddNorm(embed_dims, dropout)
    def forward(self, input):
        # 1. self Attention
        x, _ = self.self_attentions(input)

        # 2.  AddNorm
        x_normed = self.norms(input, x)

        # 3. ffn
        x_ffn = self.ffns(x_normed)

        return x_ffn

class MultiModalCoAttention(nn.Module):
    """


    """
    def __init__(self, heads, embed_dims,  dropout: float = 0.1):
        super().__init__()
        self.modality_blocks = nn.ModuleList([ModalityBlock(heads, embed_dims, dropout) for _ in range(3)])
        self.co_attention = nn.ModuleList([CoAttention(heads, dropout)for _ in range(3)])
        self.add_norm = nn.ModuleList([AddNorm(embed_dims, dropout) for _ in range(3)])

    # change forward function to account for not just stacking up the inputs
    def forward(self, text, audio, video):

        # modality separated steps 1 through 3 (SelfAttention):

        t = self.modality_blocks[0](text)
        a = self.modality_blocks[1](audio)
        v = self.modality_blocks[2](video)

        # 4.  Co-Attention

        # Triangle Scheme with kv, combined from both other modalities
        # Q: t - KV: av
        av_kv = torch.cat([a,v], dim=1)
        t_attended = CoAttention(ff_t, av_kv)
        # Q: a - KV: tv
        tv_kv = torch.cat([t,v], dim=1)
        a_attended = CoAttention(ff_a, tv_kv)
        # Q: v - KV: at
        ta_kv = torch.cat([a,t], dim=1)
        v_attended = CoAttention(ff_v, at_kv)


        # 5. AddNorm
        t_output = self.add_norm[0](ff_t, t_attended)
        a_output = self.add_norm[1](ff_a, a_attended)
        v_output = self.add_norm[2](ff_v, v_attended)

        return t_output, a_output, v_output

# ======

class CrossAttention(nn.Module):
    pass


if __name__ == '__main__':
    # dummy data - make up reasonable shapes
    batch_size = 2
    text_len = 20
    audio_len = 50
    video_len = 100
    embed_dim = 512

    text = torch.randn(batch_size, text_len, embed_dim)
    audio = torch.randn(batch_size, audio_len, embed_dim)
    video = torch.randn(batch_size, video_len, embed_dim)

    # instantiate
    model = MultiModalCoAttention(heads=8, embed_dims=embed_dim)

    # forward pass
    t_out, a_out, v_out = model(text, audio, video)

    print(t_out.shape)  # should be (2, 20, 512)
    print(a_out.shape)  # should be (2, 50, 512)
    print(v_out.shape)  # should be (2, 100, 512)