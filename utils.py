# -*- coding: utf-8 -*-
"""
@Time    : 2021/4/17 16:08
@Author  : SinGaln
"""
import json
import torch
import random
import logging
import numpy as np
import torch.nn as nn
from transformers import BertTokenizer

# load config
# def load_config():
#     return json.loads(args.config_path)

# 加载tokenizer
def load_tokenizer(args):
    return BertTokenizer.from_pretrained(args.vocab_path)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.use_cuda:
        torch.cuda.manual_seed(args.seed)

def init_logger(args):
    """
    将日志输出到日志文件和控制台
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(
        filename=args.log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 创建一个handler，用于将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger

def compute_accuracy(args, outputs, labels, device):
    # 计算平均loss和准确率
    logits = outputs
    # print("logits", logits, logits.size())
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        # print("shift_logits", shift_logits, shift_logits.size())
        shift_labels = labels[..., 1:].contiguous().to(device)
        # print("shift_labels", shift_labels, shift_labels.size())
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss(ignore_index=args.ignore_index, reduction="sum")  # 忽略padding部分,对其他部分进行loss累加
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        _, preds = shift_logits.max(dim=-1)  # preds表示对应的prediction_score预测出的token在voca中的id。维度为[batch_size,token_len]

        # 对非pad_id的token的loss进行求平均，且计算出预测的准确率
        not_ignore = shift_labels.ne(args.ignore_index)  # 进行非运算，返回一个tensor，若targets_view的第i个位置为pad_id，则置为0，否则为1
        num_targets = not_ignore.long().sum().item()  # 计算target中的非pad_id的数量

        correct = (shift_labels == preds) & not_ignore  # 计算model预测正确的token的个数，排除pad的tokne
        correct = correct.float().sum()

        accuracy = correct / num_targets
        loss = loss / num_targets
        return loss, accuracy