from collections import OrderedDict
import os
import sys
from typing import Dict
import typing
import torch

if '-h' in sys.argv or '--help' in sys.argv:
    print(f'Usage: python3 {sys.argv[0]} [--use-gpu] <lora_alpha> <base_model.pth> <lora_checkpoint.pth> <output.pth>')

device = 'cpu'
base_model, output = sys.argv[1], sys.argv[2]


with torch.no_grad():
    w: Dict[str, torch.Tensor] = torch.load(base_model, map_location='cpu')
    del w["emb.weight"]
    del w["head.weight"]
    output_w: typing.OrderedDict[str, torch.Tensor] = OrderedDict()
    # merge LoRA weights
    keys = list(w.keys())
    for k in keys:
        if k.endswith('.weight'):
            prefix = k[:-len('.weight')]
            lora_A = prefix + '.lora_A'
            lora_B = prefix + '.lora_B'
            if lora_A in keys:
                assert lora_B in keys
                output_w[lora_A] = w[lora_A].clone()
                output_w[lora_B] = w[lora_B].clone()
                print(f'extracting {k}')
                del w[k]
                del w[lora_A]
                del w[lora_B]
                continue
        if 'lora' not in k:
            print(f'retaining {k}')
            output_w[k] = w[k].clone()
            del w[k]
    torch.save(output_w, output)
