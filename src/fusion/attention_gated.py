import numpy as np
import torch.nn.functional as F
from torch._inductor import sizevars

from src.utils import l2_normalize
from co_sim_gated import _align_to_common_dim



# idea

"""
A) Text double weighted - co attention twice
1. Query Text - Key Video [TV] / Query Text - Key Audio [TA]
2. Query TA - Key TV

B) concat over the two atention embeddings

1. Query Text - Key Video [TV] / Query Text - Key Audio [TA] / Query Audio - Key Video [VA]
2. concat TV TA VA (or other way to combine these?)

"""
import numpy as np

def attention_weight(query: np.ndarray, key: np.ndarray, value: np.ndarray) -> np.ndarray:
    d_k = key.shape[-1]
    scores = (query @ key.T) / np.sqrt(d_k)
    e_x =  np.exp(scores-np.max(scores))
    softmax = e_x / e_x.sum(axis=0)
    return softmax @ value

def co_attention_gated_concatenation_multimodal(
        text_vectors: np.ndarray,
        audio_vectors: np.ndarray,
        visual_vectors: np.ndarray,
) -> np.ndarray:
    if len(text_vectors) != len(audio_vectors) or len(text_vectors) != len(visual_vectors):
        raise ValueError("Text, audio, and visual vectors must have the same segment count")

    text_n = l2_normalize(text_vectors)
    audio_n = l2_normalize(audio_vectors)
    visual_n = l2_normalize(visual_vectors)
    t_cut, a_cut, v_cut = _align_to_common_dim(text_n, audio_n, visual_n)

    # Attention Mechanism:
    # Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

    #  weighing back and front so both modalities are weighted by another
    a_weighted_t = attention_weight(t_cut, a_cut, a_cut)
    t_weighted_a = attention_weight(a_cut, t_cut, t_cut)

    v_weighted_t = attention_weight(t_cut, v_cut, v_cut)
    t_weighted_v = attention_weight(v_cut, t_cut, t_cut)

    a_weighted_v= attention_weight(v_cut, a_cut, a_cut)
    v_weighted_a = attention_weight(a_cut, v_cut, v_cut)

    # combine weights for modality

    a_weighted = a_weighted_v * a_weighted_t
    t_weighted = t_weighted_a * t_weighted_v
    v_weighted = v_weighted_a * v_weighted_t
    tav_weighted = a_weighted * v_weighted * t_weighted

    fused = np.concatenate(
        [a_weighted,
         t_weighted,
         v_weighted,
         tav_weighted
         ],
        axis=1)

    return l2_normalize(fused)


def cross_attention_gated_concatenation_multimodal(
            text_vectors: np.ndarray,
            audio_vectors: np.ndarray,
            visual_vectors: np.ndarray,
    ) -> np.ndarray:
        if len(text_vectors) != len(audio_vectors) or len(text_vectors) != len(visual_vectors):
            raise ValueError("Text, audio, and visual vectors must have the same segment count")


if __name__ == '__main__':
    segments = 2
    a = np.random.randn(segments, 3)
    v = np.random.randn(segments, 4)
    t = np.random.randn(segments, 2)
    print(a,t,v)
    print(co_attention_gated_concatenation_multimodal(a, v, t))
