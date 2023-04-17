---
license: wtfpl
---

# 全参数微调步骤

- 预处理：需要将处理前的neox风格jsonl文件放进datasource文件夹，命令参考preprocess_data.sh。不过目前已经处理了一版（去除R18内容）放进了data.
- 微调：命令参考train.sh，训练需要在RWKV-v4neo文件夹下进行、
- 推理：用ChatRWKV，或者text-generation-webui载入模型


# 胡乱记录一下

- 把模型放到pretrained_models文件夹下，或者自己参考train.sh里改改
- megatron、preprocess_data。py来自https://github.com/EleutherAI/gpt-neox
- preprocess_data.sh运行命令来自https://github.com/BlinkDL/RWKV-LM#training--fine-tuning，部分修改。
- RWKV-v4neo来自https://github.com/BlinkDL/RWKV-LM
- 训练命令参考https://www.bilibili.com/read/cv22445881，部分修改。
- 推理暂时采用temp0.7 top_p 1 