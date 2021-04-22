# coding=utf-8
# @Time:2021/4/199:09
# @author: SinGaln

import argparse
from trainer import Trainer
from data_loader import load_and_cache_examples
from utils import init_logger,set_seed, load_tokenizer


def main(args):
    init_logger(args)
    set_seed(args)
    tokenizer = load_tokenizer(args)

    train_dataset = load_and_cache_examples(args, tokenizer)

    trainer = Trainer(args, train_dataset)

    if args.do_train:
        trainer.train()

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", default="chat", required=True, type=str, help="The name of the task!")
    parser.add_argument("--data_dir", default="./data", required=True, type=str, help="The data dir of inputs.")
    parser.add_argument("--seed", default=1234, type=int, required=True, help="Seed random for initialization.")
    parser.add_argument("--train_batch_size", default=32, required=True, type=int, help="Batch sizing for training.")
    parser.add_argument("--max_seq_len", default=300, type=int, required=True, help="The maximum total input sequence length after token.")
    parser.add_argument("--learning_rate", default=5e-3, type=float, required=True, help="The initial learning rate for train.")
    parser.add_argument("--train_epochs", default=5, type=int, required=True, help="Total number of training epochs to perform.")
    parser.add_argument('--device', default='', type=str, required=False, help='设置使用哪些显卡')
    parser.add_argument('--use_cuda', action='store_true', help='不使用GPU进行训练')
    parser.add_argument('--model_config', default='config/config.json', type=str, required=False,
                        help='选择模型参数')
    parser.add_argument('--vocab_path', default='data/vocab.txt', type=str, required=False, help='选择词库')
    parser.add_argument('--train_raw_path', default='data/train.txt', type=str, required=False, help='原始训练语料')
    parser.add_argument('--train_tokenized_path', default='data/train_tokenized.txt', type=str,
                        required=False,
                        help='将原始训练语料tokenize之后的数据的存放位置')
    parser.add_argument('--log_path', default='data/training.log', type=str, required=False, help='训练日志存放位置')
    parser.add_argument('--warmup_steps', default=2000, type=int, required=False, help='warm up步数')
    parser.add_argument('--log_step', default=1000, type=int, required=False, help='多少步汇报一次loss')
    parser.add_argument('--gradient_accumulation', default=1, type=int, required=False, help='梯度积累')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--dialogue_model_output_path', default='dialogue_model/', type=str, required=False,
                        help='对话模型输出路径')
    parser.add_argument('--pretrained_model', default='', type=str, required=False, help='预训练的GPT2模型的路径')
    parser.add_argument('--writer_dir', default='tensorboard_summary/', type=str, required=False, help='Tensorboard路径')
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--config_path", default="./config/config.json", help="Config params for path.")
    parser.add_argument("--max_steps", default=-1, type=int, help="")
    parser.add_argument("--weight_decay", default=0.001, type=float, help="")
    parser.add_argument("--ignore_index", default=0, type=int, help="")
    parser.add_argument("--multi_gpu", action="store_true")
    args = parser.parse_args()

    main(args)

# python main.py --task chat --data_dir ./data --seed 1234 --train_batch_size 2 --max_seq_len 300 --learning_rate 5e-5 --train_epochs 2 --do_train