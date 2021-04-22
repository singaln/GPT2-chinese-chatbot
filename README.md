##### 项目介绍：

​		本项目是基于GPT2实现的一个中文闲聊模型，其中的模型代码是自己编写的，可能不是很适合，需要的小伙伴可以直接调用transformers库的GPT2Model来在trainer.py文件的模型加载部分进行替换。

##### 项目数据：

​		项目的数据为一些自己抓取的聊天数据，格式如下：

```markdown
你吃了吗？
吃过了，吃的糖醋排骨，你呢
我吃的是麻辣小龙虾

手机欠费了怎么办？
交话费啊
去哪里才能交话费呢
去相应的营业厅啊
```

​		数据格式就是一段对话，不同的对话间使用空行隔开。

##### 运行：

```python
# python main.py --task chat --data_dir ./data --seed 1234 --train_batch_size 2 --max_seq_len 300 --learning_rate 5e-5 --train_epochs 2 --do_train
```

