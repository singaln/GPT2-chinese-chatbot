# -*- coding: utf-8 -*-
"""
@Time    : 2021/4/17 16:28
@Author  : SinGaln
"""
import os
import torch
import logging
from tqdm import tqdm
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)

class DataProcess(object):
    def __init__(self, args):
        self.args = args
        self.data_file = "./data/train.txt"

    @classmethod
    def _read_data_file(cls, input_file):
        logger.info("tokenizing raw data,raw data path:{}".format(input_file))
        with open(input_file, 'rb') as f:
            data = f.read().decode("utf-8")
        if "\r\n" in data:
            train_data = data.split("\r\n\r\n")
        else:
            train_data = data.split("\n\n")
        logger.info("there are {} dialogue in raw dataset".format(len(train_data)))
        return train_data

    def get_examples(self, tokenizer):
        context = []
        train_data = self._read_data_file(self.data_file)
        for dialogue_index, dialogue in enumerate(tqdm(train_data)):
            utterances = dialogue.split("\n")
            dialogue_ids = [tokenizer.cls_token_id]  # 每个dialogue以[CLS]开头
            for utterance in utterances:
                dialogue_ids.extend([tokenizer.convert_tokens_to_ids(word) for word in utterance])
                dialogue_ids.append(tokenizer.sep_token_id)  # 每个utterance之后添加[SEP]，表示utterance结束
            # 对超过n_ctx的长度进行截断,否则GPT2模型会报错
            if len(dialogue_ids) > self.args.max_seq_len:
                dialogue_ids = dialogue_ids[:self.args.max_seq_len]
                print("dialogue", len(dialogue_ids))
            else:
                dialogue_ids = dialogue_ids + ([0] * (self.args.max_seq_len - len(dialogue_ids)))
            context.append(dialogue_ids)
        logger.info("finish processing for raw data!")
        return context


processors = {
    "chat": DataProcess
}
def load_and_cache_examples(args, tokenizer):
    processor = processors[args.task](args)

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        'cached_train_{}_{}'.format(
            args.task,
            args.max_seq_len
        )
    )

    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        # Load data features from dataset file
        logger.info("Creating features from dataset file at %s", args.data_dir)
        features = processor.get_examples(tokenizer)

        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor(features, dtype=torch.long)

    dataset = TensorDataset(all_input_ids)
    return dataset