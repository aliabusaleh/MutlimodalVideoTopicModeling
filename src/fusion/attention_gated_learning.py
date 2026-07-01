import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.fusion.co_sim_gated import _align_to_common_dim

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
            nn.Linear(embed_dims, embed_dims* 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dims * 4, embed_dims)
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
            embed_dim=embed_dims,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )

        self.norm1 = AddNorm(embed_dims)
        self.linear = nn.Linear(embed_dims, embed_dims)
        self.norm2 = AddNorm(embed_dims)

    def forward(self, i):
        x, attention_weights = self.attention(i, i, i)
        x_norm = self.norm1(i, x)
        x_linear = self.linear(x_norm)
        x_normed = self.norm2(x_norm, x_linear)
        return x_normed, attention_weights


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


# ================ CO ATTENTION ===========================
class CoAttention(nn.Module):
    def __init__(self, num_heads, embed_dims):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dims,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dims)


    def forward(self, query, key_value):
        attn_output, attn_weights = self.attention(query, key_value, key_value)
        query = self.norm(query + attn_output)
        return query, attn_weights


class MultiModalCoAttention(nn.Module):
    """
    """
    def __init__(self, heads,  dropout: float = 0.1):
        super().__init__()
        self.heads = heads
        self.dropout = dropout
        self._initialized = False
        
    def _build(self, embed_dims):
        self.embed_dims = 512
        self.modality_blocks = nn.ModuleList([ModalityBlock(self.heads, embed_dims, self.dropout) for _ in range(3)])
        self.co_attention = nn.ModuleList([CoAttention(self.heads, embed_dims)for _ in range(3)])
        self.add_norm = nn.ModuleList([AddNorm(embed_dims, self.dropout) for _ in range(3)])
        

        # learned fusion layer
        self.fusion_mlp = nn.Sequential(
            nn.Linear(3 * embed_dims, 2 * embed_dims),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2 * embed_dims, embed_dims)
        )
        self._initialized = True


    # change forward function to account for not just stacking up the inputs
    def forward(self, text, audio, video):
        #text, audio, video = _align_to_common_dim(text, audio, video)
        try:
            if not self._initialized:

                self._build(text.shape[-1])
            print("after initialize in forward")
            print(text.shape, audio.shape, video.shape)  # all should match

            # modality separated steps 1 through 3 (SelfAttention):
            text = text.unsqueeze(1)
            audio = audio.unsqueeze(1)
            video = video.unsqueeze(1)
            print("after unsqueeze")
            print(text.shape, audio.shape, video.shape)  # all should match

            t = self.modality_blocks[0](text)
            a = self.modality_blocks[1](audio)
            v = self.modality_blocks[2](video)

            # 4.  Co-Attention

            # Triangle Scheme with kv, combined from both other modalities
            # Q: t - KV: av
            av_kv = torch.cat([a,v], dim=1)
            t_attended, _ = self.co_attention[0](t, av_kv)
            # Q: a - KV: tv
            tv_kv = torch.cat([t,v], dim=1)
            a_attended, _ = self.co_attention[1](a, tv_kv)
            # Q: v - KV: at
            ta_kv = torch.cat([a,t], dim=1)
            v_attended, _ = self.co_attention[2](v, ta_kv)


            # 5. AddNorm
            t_output = self.add_norm[0](t, t_attended)
            a_output = self.add_norm[1](a, a_attended)
            v_output = self.add_norm[2](v, v_attended)

            fused = torch.cat(
            [t_output,
             a_output,
             v_output],
            dim=-1
            )

            print(t_output.shape, a_output.shape, v_output.shape)
            print(t_output.mean(dim=1).shape)
            print(fused.shape)
            fusion = self.fusion_mlp(fused)

            return fusion.squeeze(1)
        except Exception:
            traceback.print_exc()


# ====== CROSS ATTENTION ==============

# TODO

class CrossAttention(nn.Module):
    pass

def contrastive_loss(embeddings, temperature=0.07):
    """
    NT xent contrastive loss, for positive pairs (same image, text and audio) and negative pairs
    :param embeddings:
    :param temperature:
    :return:
    """
    embeddings = F.normalize(embeddings, dim=-1)
    # making a similarity matrix
    sim_matrix = embeddings @ embeddings.T / temperature
    # diagonal: each video embeds compared with embeddings from same video
    # non-fiagonal: compared to a different video
    labels = torch.arange(sim_matrix.size(0), device=embeddings.device)

    #cross entropy loss
    loss = F.cross_entropy(sim_matrix, labels)
    return loss

def run(text_vectors, audio_vectors, video_vectors, epochs=10, heads=8):
    try:
        model = MultiModalCoAttention(heads=heads)
        #print(text_vectors.shape, audio_vectors.shape, video_vectors.shape)
        #print(text_vectors[:1].shape, audio_vectors[:1].shape, video_vectors[:1].shape)
        dummy = model(text_vectors[:1], audio_vectors[:1], video_vectors[:1])

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        for epoch in range(epochs):
            optimizer.zero_grad()
            fused = model(text_vectors, audio_vectors, video_vectors)
            contrast_loss = contrastive_loss(fused)
            contrast_loss.backward()
            optimizer.step()
        return model
    except Exception:
        traceback.print_exc()

