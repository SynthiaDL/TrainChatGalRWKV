---
license: wtfpl
---

## 全参数微调步骤

- 预处理：需要将处理前的neox风格jsonl文件放进`datasource`文件夹，命令参考`preprocess_data.sh`。目前已经处理了一版（去除R18内容）放进了`data`文件夹
- 微调：命令参考`train.sh`，训练需要在`RWKV-v4neo`文件夹下进行
- 推理：用ChatRWKV，或者text-generation-webui载入模型

## 胡乱记录一下

- 把模型放到`pretrained_models`文件夹下，或者自己参考`train.sh`里改改
- `megatron`、`preprocess_data.py`来自 https://github.com/EleutherAI/gpt-neox
- `preprocess_data.sh`运行命令来自 https://github.com/BlinkDL/RWKV-LM#training--fine-tuning ，部分修改
- `RWKV-v4neo`来自 https://github.com/BlinkDL/RWKV-LM
- 训练命令参考 https://www.bilibili.com/read/cv22445881 ，部分修改
- 推理暂时采用`temp = 0.7, top_p = 1`
- train.sh参数解释
- ```
这是一个用于训练深度学习模型的命令行脚本。以下是参数的解释：

--load_model：指定预训练模型的路径，用作训练的初始模型。
--wandb：设置 Weights & Biases（一个用于深度学习实验跟踪的平台）的项目名称。
--data_file：指定训练数据文件的路径。
--data_type：设置训练数据的类型。
--vocab_size：设置词汇表大小。
--ctx_len：设置上下文长度。
--accumulate_grad_batches：设置梯度累积批次，用于梯度累积优化。
--epoch_steps：设置每个epoch的训练步数。
--epoch_count：设置训练的总epoch数。
--epoch_save：设置每隔多少个epoch保存一次模型。
--micro_bsz：设置微批次大小。
--n_layer：设置模型的层数。
--n_embd：设置模型的嵌入维度。
--pre_ffn：设置前馈神经网络（FFN）的预处理（0表示不使用）。
--head_qk：设置注意力头的查询和键（0表示不使用）。
--lr_init：设置初始学习率。
--lr_final：设置最终学习率。
--warmup_steps：设置学习率预热步数。
--beta1：设置 Adam 优化器的 beta1 参数。
--beta2：设置 Adam 优化器的 beta2 参数。
--adam_eps：设置 Adam 优化器的 epsilon 参数。
--accelerator：设置加速器类型（例如 GPU）。
--devices：设置使用的设备数量。
--precision：设置计算精度（例如 bfloat16）。
--strategy：设置训练策略（例如 deepspeed_stage_2_offload）。
--grad_cp：梯度检查点标志。
```