# -*- coding: utf-8 -*-
"""
@Time    : 2021/4/17 17:18
@Author  : SinGaln
"""
import os
import logging
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torch
from config import Config
from torch.utils.data import DataLoader, RandomSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from model import GPT2_model

from utils import compute_accuracy

logger = logging.getLogger(__name__)
config = Config()

class Trainer(object):
    def __init__(self, args, train_dataset=None):
        self.args = args
        self.train_dataset = train_dataset

        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        # self.pad_token_label_id = args.ignore_index
        self.config = config
        self.model = GPT2_model.GPT2Model(self.config)

        # GPU or CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)

        if self.args.max_steps > 0:
            total_steps = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation) + 1
        else:
            total_steps = len(train_dataloader) // self.args.gradient_accumulation * self.args.train_epochs
        logger.info('total training steps = {}'.format(total_steps))

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=total_steps)

        logger.info('starting training')
        # 用于统计每次梯度累计的loss
        running_loss = 0
        # 统计一共训练了多少个step
        overall_step = 0
        # 记录tensorboardX
        # tb_writer = SummaryWriter(log_dir=self.args.writer_dir)
        # 记录 out of memory的次数
        oom_time = 0
        # 开始训练
        for epoch in range(self.args.train_epochs):
            epoch_start_time = datetime.now()
            for steps, input_ids in enumerate(train_dataloader):
                # print("steps", steps)
                # print("inputs_ids", input_ids)
                # 注意：GPT2模型的forward()函数，是对于给定的context，生成一个token，而不是生成一串token
                # GPT2Model的输入为n个token_id时，输出也是n个hidden_state，使用第n个hidden_state预测第n+1个token
                input_ids = input_ids[0].to(self.device)
                # 解决在运行过程中，由于显存不足产生的cuda out of memory的问题
                try:
                    outputs = self.model(input_ids)
                    loss, accuracy = compute_accuracy(self.args, outputs, labels=input_ids, device=self.device)
                    logger.info("loss: {}  acc:{}".format(loss, accuracy))
                    # if self.args.multi_gpu:
                    #     loss = loss.mean()
                    #     accuracy = accuracy.mean()
                    # if self.args.gradient_accumulation > 1:
                    #     loss = loss / self.args.gradient_accumulation
                    #     accuracy = accuracy / self.args.gradient_accumulation
                    loss.backward()
                    # 梯度裁剪解决的是梯度消失或爆炸的问题，即设定阈值
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    # 进行一定step的梯度累计之后，更新参数
                    if (steps + 1) % self.args.gradient_accumulation == 0:
                        running_loss += loss.item()
                        # 更新参数
                        optimizer.step()
                        # 清空梯度信息
                        optimizer.zero_grad()
                        # 进行warm up
                        scheduler.step()
                        overall_step += 1
                        # 更新日志与tnesorboardX信息
                        if (overall_step + 1) % self.args.log_step == 0:
                            logger.info(
                                "batch {} of epoch {}, loss {}, accuracy {}".format(steps + 1, epoch + 1, loss,
                                                                                    accuracy))
                            # tb_writer.add_scalar('loss', loss.item(), overall_step)
                except RuntimeError as exception:
                    if "out of memory" in str(exception):
                        oom_time += 1
                        logger.info("WARNING: ran out of memory,times: {}".format(oom_time))
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                    else:
                        logger.info(str(exception))
                        raise exception
            self.save_model(epoch, epoch_start_time)

    def save_model(self, epoch, epoch_start_time):
        logger.info('saving model for epoch {}'.format(epoch + 1))

        model_path = os.path.join(self.args.dialogue_model_output_path, 'model_epoch{}'.format(epoch + 1))
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(model_path)
        logger.info('epoch {} finished'.format(epoch + 1))
        epoch_finish_time = datetime.now()
        logger.info('time for one epoch: {}'.format(epoch_finish_time - epoch_start_time))
        logger.info('training finished')
