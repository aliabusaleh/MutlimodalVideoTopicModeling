import torch
import torch.nn as nn
"""
https://arturmagalhaes.com/research/python/2025/10/23/attention-mechanisms.html

https://github.com/nestor-sun/mcoattention
"""
class CoAttentionLayer(nn.Module):
    """
    3 Modalities as "triangle" with undirected edges:
        => compute attention in any direction between the modalities simultaneously with coupled parameters

    Co-Attention:
        Modality A uses its own Q, but attends to the other modalities K.
        "Coupled" as: each modalities K and V serves as Context for the other modalities Q.
    """
    def __init__(
            self,
            embed_dim: int,
            number_heads: int = 12 # TODO Adjust this if needed
    ):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        self.num_heads = number_heads
        self.head_dim = embed_dim // number_heads
        self.scale = self.head_dim ** -0.5 # scale down the scores for the soft max function


        # seperate Q, and combined KV for each modality

        self.text_q = nn.Linear(embed_dim, embed_dim)
        self.audio_q = nn.Linear(embed_dim, embed_dim)
        self.video_q = nn.Linear(embed_dim, embed_dim)

        self.text_kv = nn.Linear(embed_dim, embed_dim*2)
        self.audio_kv = nn.Linear(embed_dim, embed_dim *2)
        self.video_kv = nn.Linear(embed_dim, embed_dim*2)

        self.text_out = nn.Linear(embed_dim, embed_dim)
        self.audio_out = nn.Linear(embed_dim, embed_dim)
        self.video_out = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(0.1)

    def forward(self, text_features, audio_features, video_features):
        batch_size = text_features.shape[0]
        text_len = text_features.shape[1]
        audio_len = audio_features.shape[1]
        video_len = video_features.shape[1]

        # 1. features
        text_q = self.text_q(text_features).view(batch_size, text_len, self.num_heads, self.head_dim).transpose(1, 2)
        text_k, text_v = self.text_kv(text_features).chunk(2, dim=-1)
        text_k = text_k.view(batch_size, text_len, self.num_heads, self.head_dim).transpose(1, 2)
        text_v = text_v.view(batch_size, text_len, self.num_heads, self.head_dim).transpose(1, 2)

        audio_q = self.audio_q(audio_features).view(batch_size, audio_len, self.num_heads, self.head_dim).transpose(1, 2)
        audio_k, audio_v = self.audio_kv(audio_features).chunk(2, dim=-1)
        audio_k = audio_k.view(batch_size, audio_len, self.num_heads, self.head_dim).transpose(1, 2)
        audio_v = audio_v.view(batch_size, audio_len, self.num_heads, self.head_dim).transpose(1, 2)


        video_q = self.video_q(video_features).view(batch_size, video_len, self.num_heads, self.head_dim).transpose(1, 2)
        video_k, video_v = self.video_kv(video_features).chunk(2, dim=-1)
        video_k = video_k.view(batch_size, video_len, self.num_heads, self.head_dim).transpose(1, 2)
        video_v = video_v.view(batch_size, video_len, self.num_heads, self.head_dim).transpose(1, 2)

        # co attention 3 way system (like a triangle)
        # a<->t , v<->t, a<->v
        t_to_a_scores = torch.matmul(text_q, audio_k.transpose(-2, -1)) * self.scale
        a_to_t_scores = torch.matmul(audio_q, text_k.transpose(-2, -1)) * self.scale

        v_to_t_scores = torch.matmul(video_q, text_k.transpose(-2, -1)) * self.scale
        t_to_v_scores = torch.matmul(text_q, video_k.transpose(-2, -1)) * self.scale

        a_to_v_scores = torch.matmul(audio_q, video_k.transpose(-2, -1)) * self.scale
        v_to_a_scores = torch.matmul(video_q, audio_k.transpose(-2, -1)) * self.scale

        # 2. attention (and dropout)
        t_to_a_attn_weights = self.dropout(F.softmax(t_to_a_scores, dim=-1))
        a_to_t_attn_weights = self.dropout(F.softmax(a_to_t_scores, dim=-1))

        v_to_t_attn_weights = self.dropout(F.softmax(v_to_t_scores, dim=-1))
        t_to_v_attn_weights = self.dropout(F.softmax(t_to_v_scores, dim=-1))

        a_to_v_attn_weights = self.dropout(F.softmax(a_to_v_scores, dim=-1))
        v_to_a_attn_weights = self.dropout(F.softmax(v_to_a_scores, dim=-1))

        # attend as weighted sum of the other modalities
        t_attended = torch.matmul(t_to_a_attn_weights, audio_v) + torch.matmul(t_to_v_attn_weights, video_v)
        a_attended = torch.matmul(a_to_v_attn_weights, video_v) + torch.matmul(a_to_t_attn_weights, text_v)
        v_attended = torch.matmul(v_to_a_attn_weights, audio_v) + torch.matmul(v_to_t_attn_weights, text_v)

        t_attended = t_attended.transpose(1, 2).contiguous().view(batch_size, text_len, -1)
        a_attended = a_attended.transpose(1, 2).contiguous().view(batch_size, audio_len, -1)
        v_attended = v_attended.transpose(1, 2).contiguous().view(batch_size, video_len, -1)

        # 3. add original features and normalize
        t = self.norm(t_attended + text_features)
        a = self.norm(a_attended + audio_features)
        v = self.norm(v_attended + video_features)

        # 4. normalize
        t = self.norm(t)
        a = self.norm(a)
        v = self.norm(v)

        # 5. FFN

        # 6. add

        # 7. normalize

        # 8. output
        text_output = self.text_out(t_attended)
        audio_output = self.audio_out(a_attended)
        video_output = self.video_out(v_attended)

        return text_output, audio_output, video_output