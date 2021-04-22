# -*- coding: utf-8 -*-
"""
@Time    : 2021/4/17 16:09
@Author  : SinGaln
"""
"""搭建GPT2"""
import os
import math
import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm_notebook, trange
from torch.nn.modules.normalization import LayerNorm

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


# 构建线下转换层
class MLP(nn.Module):
    def __init__(self, embedding_size):
        super(MLP, self).__init__()
        self.dense_h_to_4h = nn.Linear(embedding_size, embedding_size * 4)
        self.dense_4h_to_h = nn.Linear(embedding_size * 4, embedding_size)
        self.act = nn.functional.gelu

    def forward(self, x):
        h = self.act(self.dense_h_to_4h(x))
        h2 = self.dense_4h_to_h(h)
        return h2


# 线性层测试
"""
model = Linear(768, 768*3)
x = torch.rand(1,4,768) [batch_size, seq_len, dim]
y = model(x)
print(y, y.shape) [1, 4, 2304]
"""


class Attention(nn.Module):
    def __init__(self, embedding_size, num_attention_heads, attention_dropout, residual_dropout):
        super(Attention, self).__init__()

        self.num_attention_heads = num_attention_heads
        self.size_per_head = embedding_size // num_attention_heads
        self.embedding_size = embedding_size

        self.query_key_value = nn.Linear(embedding_size, embedding_size * 3)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.resid_drop = nn.Dropout(residual_dropout)
        self.dense = nn.Linear(embedding_size, embedding_size)

    def split_heads(self, x):
        "return shape [`batch`, `head`, `sequence`, `features`]"
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.size_per_head)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_shape)

    def forward(self, x, kv_cache=None):
        self.seq_len = x.size(1)

        # self_attention
        x = self.query_key_value(x)
        q, k, v = x.split(self.embedding_size, dim=2)

        # 多头
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        cached_kv = torch.stack([k, v], dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / math.sqrt(self.size_per_head)
        attention_mask = torch.tril(torch.ones([self.seq_len, self.seq_len], dtype=torch.float32))
        # print("attention", attention_mask)
        attention_mask = attention_mask.reshape([1, 1, self.seq_len, self.seq_len])
        # print(1.0 - attention_mask)
        # print(scores * attention_mask)
        scores = scores * attention_mask - 10000.0 * (1.0 - attention_mask)
        # print(scores)
        scores = nn.Softmax(dim=1)(scores)
        scores = self.attn_drop(scores)
        y = torch.matmul(scores, v)
        y = self.merge_heads(y)
        y = self.resid_drop(self.dense(y))
        return y, cached_kv


class Block(nn.Module):
    def __init__(self, embedding_size, num_attention_heads, attention_dropout, residual_dropout):
        super(Block, self).__init__()
        self.input_layernorm = nn.LayerNorm(embedding_size, eps=1e-5)
        self.attention = Attention(embedding_size, num_attention_heads, attention_dropout, residual_dropout)
        self.post_attention_layernorm = nn.LayerNorm(embedding_size, eps=1e-5)
        self.mlp = MLP(embedding_size)

    def forward(self, x, kv_cache=None):
        # Attention + 前后的LayerNorm + 中间残差连接
        attn, cached_kv = self.attention(self.input_layernorm(x), kv_cache=kv_cache)
        x = x + attn
        z = self.post_attention_layernorm(x)

        # MLP
        z = self.mlp(z)

        # 残差连接
        x = x + z
        return x, cached_kv


class Transformer(nn.Module):
    def __init__(self,
                 layer_size,
                 embedding_size,
                 num_attention_heads,
                 attention_dropout,
                 residual_dropout):
        super(Transformer, self).__init__()

        self.layers = nn.ModuleList([Block(
            embedding_size,
            num_attention_heads,
            attention_dropout,
            residual_dropout)
            for _ in range(layer_size)])

        self.final_layernorm = nn.LayerNorm(embedding_size, eps=1e-5)

    def forward(self, x, kv_cache=None):
        # 多层 Block
        cached_kvs = []
        for i, layer in enumerate(self.layers):
            x, cached_kv = layer(
                x, kv_cache=kv_cache[i] if kv_cache is not None else None)
            cached_kvs.append(cached_kv)

        # 最终的 LayerNorm
        x = self.final_layernorm(x)

        return x, torch.stack(cached_kvs)


class GPT2Model(nn.Module):
    def __init__(self, config):
        super(GPT2Model, self).__init__()

        # 定义字符嵌入层
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size)

        # 定义位置嵌入层
        self.position_embeddings = nn.Embedding(config.block_size, config.embedding_size)

        # 定义嵌入随机丢弃层
        self.emb_drop = nn.Dropout(config.embedding_dropout)

        # 定义 Transformer Encoder
        self.transformer = Transformer(
            config.layer_size,
            config.embedding_size,
            config.num_attention_heads,
            config.attention_dropout,
            config.residual_dropout)

    def forward(self, x, kv_cache=None, use_cache=False):
        # 根据缓存确定历史输入长度
        if kv_cache is None:
            past_length = 0
        else:
            past_length = kv_cache[0][0].shape[-2]

        # 生成位置编码
        position_ids = torch.arange(past_length, x.shape[-1] + past_length, dtype=torch.int64)
        position_ids = position_ids.unsqueeze(0).expand_as(x)

        # 计算嵌入层输出
        x = self.word_embeddings(x)
        x = self.emb_drop(x + self.position_embeddings(position_ids))

        # 计算 Transformer Encoder 输出
        x, cached_kvs = self.transformer(x, kv_cache)

        # 计算解码输出
        # 解码使用的参数为字符嵌入层参数的转置
        # 相当于做一个逆运算或者可以理解为使用相同的参数进行编码和解码
        x = torch.matmul(x, self.word_embeddings.weight.transpose(-1, -2))

        # 如果使用缓存则返回输出和缓存
        if use_cache:
            return x, cached_kvs

        # 否则只返回输出
        return x