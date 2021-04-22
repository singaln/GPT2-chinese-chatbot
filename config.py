# coding=utf-8
# @Time:2021/4/1910:17
# @author: SinGaln

class Config(object):
    def __init__(self):
        self.initializer_range = 0.02
        self.layer_norm_epsilon = 1e-05
        self.n_ctx = 300
        self.embedding_size=768
        self.embedding_dropout=0.0
        self.num_attention_heads=12
        self.residual_dropout=0.0
        self.attention_dropout=0.0
        self.layer_size=10
        self.block_size=1024
        self.n_positions=300
        self.vocab_size=19020