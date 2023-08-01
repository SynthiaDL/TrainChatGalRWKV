---
license: Apache 2.0
---

# 项目简介

本项目是基于[RWKV-LM-LoRA](https://github.com/Blealtan/RWKV-LM-LoRA/)项目修改而来，用于ChatGal数据训练。由于本项目加入了一些社区内的最新解决方案，能够更好地实现训练RWKV的长上下文能力，因此也可供其它项目试用。

主要特点

- Ctrl+C退出时自动保存最新模型参数
- 规避cuda kernel造成的显存瓶颈，支持24G显存显卡全量训练RWKV 7B模型、LoRA训练RWKV 7B模型。
- 实验性的无限长上下文训练

本项目主体代码与[RWKV-LM-LoRA](https://github.com/Blealtan/RWKV-LM-LoRA/)相同，可以参考其训练配置。**提醒一下：训练LoRA模型需要先和主模型合并才能使用！（使用merge_lora.py或者merge_lora_plus.py合并）。**

由于项目屎山堆积比较重，未来代码可能会整理重构。这里仅简单介绍一下长上下文的解决方案。
RWKV模型训练过程中训练文本长度ctx_len会对显存占用有重要影响，除了会增大激活值占用的存储，也会增大wkv kernel运算过程中的临时存储开销，导致24G显存下7B模型常常无法训练4096及以上文本（即使已经开启了层间梯度检查点（Gradient Checkpointing） `--grad_cp 1`）。

## 方案1：简单降低wkv显存开销

本项目使用了修改后的wkv kernel，支持状态输入。因此一种解决方案是在执行wkv运算时，把输入序列分片成较小序列长度，依次进行wkv运算，并保持状态的正常传递。

实现上可以调整`--ctx_parts`参数，从1逐渐增大。例如当`--ctx_len 4096 --ctx_parts 4`时，模型实际使用的是1024长度的wkv kernel，避免因wkv运算造成OOM。

这种解决方案可以轻微降低显存开销，扩展最大可训练长度，但是无法解决因激活值和最后一层计算logits产生的显存开销，因此提升有限。但是本方案的优点是兼容deepspeed stage 2及offload，相对于原版RWKV-LM可以无副作用地降低显存。

## 方案2：无限长上下文训练（状态梯度检查点）

[infctx](https://github.com/Blealtan/RWKV-LM-LoRA/tree/dev-infctx)是RWKV社区内Blealtan等人实现的一种RWKV训练程序，可以实现充分长序列的训练。其原理是利用RWKV作为RNN的特点，在时间方向上进行梯度检查点，通过两次前向传播的代价换取显存降低，支持长序列训练。

开启需要给出参数`--state_cp 1`，为了让模型逐渐适应长上下文，建议开启`--initial_ctx_len 4096 --ctx_warmup_steps 200`，通过200步逐渐将训练长度从4096涨到ctx_len。

在24G单卡上使用lora训练7B模型，最大验证过可以训练128K及以上文本长度（并且显存还有大量剩余）。

由于deepspeed框架的一些问题，导致这种方案仅适用于deepspeed stage 1。对于单张24G显存的显卡，该种方案只能LoRA微调7B模型。如果要全量微调7B模型，至少需要4x80G显卡。


# 杂项

## 全参数微调步骤

- 预处理：需要将处理前的neox风格jsonl文件放进`datasource`文件夹，命令参考`preprocess_data.sh`。目前已经处理了一版（去除R18内容）放进了`data`文件夹
- 微调：命令参考`train.sh`，训练需要在`RWKV-v4neo-LoRA`文件夹下进行
- 推理：用ChatRWKV，或者text-generation-webui载入模型。

## 胡乱记录一下

- 把模型放到`pretrained_models`文件夹下，或者自己参考`train.sh`里改改
- `megatron`、`preprocess_data.py`来自 https://github.com/EleutherAI/gpt-neox
- `preprocess_data.sh`运行命令来自 https://github.com/BlinkDL/RWKV-LM#training--fine-tuning ，部分修改
- `RWKV-v4neo`来自 https://github.com/BlinkDL/RWKV-LM
- 训练命令参考 https://www.bilibili.com/read/cv22445881 ，部分修改
- 推理暂时采用`temp = 0.7, top_p = 1`
- train.sh参数解释
- load_model：指定预训练模型的路径，用作训练的初始模型。
- wandb：设置 Weights & Biases（一个用于深度学习实验跟踪的平台）的项目名称。
- data_file：指定训练数据文件的路径。
- data_type：设置训练数据的类型。
- vocab_size：设置词汇表大小。
- ctx_len：设置上下文长度。
- accumulate_grad_batches：设置梯度累积批次，用于梯度累积优化。
- epoch_steps：设置每个epoch的训练步数。
- epoch_count：设置训练的总epoch数。
- epoch_save：设置每隔多少个epoch保存一次模型。
- micro_bsz：设置微批次大小。
- n_layer：设置模型的层数。
- n_embd：设置模型的嵌入维度。
- pre_ffn：设置前馈神经网络（FFN）的预处理（0表示不使用）。
- head_qk：设置注意力头的查询和键（0表示不使用）。
- lr_init：设置初始学习率。
- lr_final：设置最终学习率。
- warmup_steps：设置学习率预热步数。
- beta1：设置 Adam 优化器的 beta1 参数。
- beta2：设置 Adam 优化器的 beta2 参数。
- adam_eps：设置 Adam 优化器的 epsilon 参数。
- accelerator：设置加速器类型（例如 GPU）。
- devices：设置使用的设备数量。
- precision：设置计算精度（例如 bfloat16）。
- strategy：设置训练策略（例如 deepspeed_stage_2_offload）。
- grad_cp：层梯度检查点标志。
- state_cp: 状态梯度检查点标志。开启后，将会大幅减少长上下文的显存占用。
- initial_ctx_len: 初始训练上下文长度。配合ctx_warmup_steps使用
- ctx_warmup_steps: 上下文warmup步数，训练过程中会逐渐从initial_ctx_len涨到ctx_len
- ctx_parts: 上下文切片数量。训练过程中会使用ctx_len/ctx_parts长度的WKV算子。例如ctx_len=8192,ctx_parts=8，则会使用1024长度的WKV算子。WKV算子的长度越短，占用显存越小，但是可能需要更多的计算时间。
