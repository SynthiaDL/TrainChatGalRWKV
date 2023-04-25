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