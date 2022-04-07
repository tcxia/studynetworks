from collections import namedtuple
import torch
import torch.nn as nn


from linformer import LinformerSelfAttention


linformerSetting = namedtuple('LinformerSettings', ['k'])


def exists(val):
    return val is not None

class LinearAttentionTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        max_seq_len,
        heads=8,
        dim_head=None,
        bucket_size=64,
        causal=False,
        ff_chunks=1,
        ff_glu=False,
        attn_layer_dropout=0.0,
        attn_dropout=0.0,
        reversible=False,
        blindspot_size=1,
        n_local_attn_heads=0,
        local_attn_window_size=128,
        receives_context=False,
        attend_axially=False,
        pkm_layers=tuple(),
        pkm_num_keys=128,
        linformer_settings=None,
        context_linformer_settings=None,
        shift_tokens=False,
    ) -> None:
        super().__init__()
        assert not (causal and exists(linformer_settings)), 'Linfformer self attention layer can only be used for non-causal networks'
        assert not exists(linformer_settings) or isinstance(linformer_settings, linformerSetting), 'Linformer self-attention settings must a LinformerSetting namedtuple'