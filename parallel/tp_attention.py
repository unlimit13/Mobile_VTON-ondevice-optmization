"""Tensor-parallel (Megatron-style) sharding for the Mobile-VTON Attention module.

column-parallel to_q/to_k/to_v (each rank keeps its head slice; no comm) +
row-parallel to_out (each rank computes a partial output; one all_reduce sums
them). Per attention: exactly one all_reduce of [B, S, out_dim].

    shard_attention_(attn, rank, world)       # slice weights in place
    attn.set_processor(TPAttnProcessor(group)) # processor that all_reduces to_out

MVP limits (see PROGRESS.md): standard MHA only (kv_heads == heads), no rope.
"""
import torch
import torch.distributed as dist
import torch.nn.functional as F


def _slice_rows(linear, start, end):
    """Keep output features [start:end] of an nn.Linear (column-parallel)."""
    linear.weight = torch.nn.Parameter(linear.weight.data[start:end].clone())
    if linear.bias is not None:
        linear.bias = torch.nn.Parameter(linear.bias.data[start:end].clone())
    linear.out_features = end - start


def _slice_cols(linear, start, end, keep_bias):
    """Keep input features [start:end] of an nn.Linear (row-parallel).

    Bias is applied to the FULL output, so it must be counted once across ranks:
    keep it on rank 0, zero it elsewhere -> after all_reduce the bias is correct.
    """
    linear.weight = torch.nn.Parameter(linear.weight.data[:, start:end].clone())
    if linear.bias is not None and not keep_bias:
        linear.bias = torch.nn.Parameter(torch.zeros_like(linear.bias.data))
    linear.in_features = end - start


def shard_attention_(attn, rank, world):
    """In-place TP shard of one Attention module across `world` ranks."""
    H = attn.heads
    assert H % world == 0, f"heads {H} not divisible by world {world}"
    head_dim = attn.inner_dim // H
    kv_heads = attn.inner_kv_dim // head_dim
    assert kv_heads == H, f"MVP supports MHA only (kv_heads={kv_heads}, heads={H})"
    local = H // world
    s, e = rank * local * head_dim, (rank + 1) * local * head_dim
    _slice_rows(attn.to_q, s, e)
    _slice_rows(attn.to_k, s, e)
    _slice_rows(attn.to_v, s, e)
    _slice_cols(attn.to_out[0], s, e, keep_bias=(rank == 0))
    # the processor reshapes by attn.heads -> set to this rank's local head count
    attn.heads = local
    attn.inner_dim = local * head_dim
    attn.inner_kv_dim = local * head_dim
    return attn


class TPAttnProcessor:
    """AttnProcessor2_0 + row-parallel to_out all_reduce. Use after shard_attention_."""

    def __init__(self, group=None):
        self.group = group  # gloo process group (None -> default WORLD)

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, image_rotary_emb=None,
                 *args, **kwargs):
        if image_rotary_emb is not None:
            raise NotImplementedError("rope not supported in TP MVP")

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            b, c, h, w = hidden_states.shape
            hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)

        batch_size, seq, _ = (
            hidden_states.shape if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, seq, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1,
                                                 attention_mask.shape[-1])
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        head_dim = query.shape[-1] // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim).to(query.dtype)

        # row-parallel: each rank produces a PARTIAL output, sum across ranks.
        hidden_states = attn.to_out[0](hidden_states)
        dist.all_reduce(hidden_states, op=dist.ReduceOp.SUM, group=self.group)
        hidden_states = attn.to_out[1](hidden_states)  # dropout (eval: no-op)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(b, c, h, w)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor
