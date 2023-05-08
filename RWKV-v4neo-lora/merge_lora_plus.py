from collections import OrderedDict
import os
import sys
from typing import Dict
import typing
import torch
import argparse
import pdb


parser = argparse.ArgumentParser(description='Merge LoRA-only slim checkpoint into the main weights')
parser.add_argument('--use-gpu', action='store_true', help='use GPU')
parser.add_argument('lora_alpha', type=float, help='lora alpha')
parser.add_argument('base_model', type=str, help='base model path')
parser.add_argument('lora', type=str, help='lora checkpoint path')
parser.add_argument('output', type=str, help='output path')
parser.add_argument('--layer_filter',type=str,default=None,help='layer filter. Default merge all layer. Example: "25-31"')
args = parser.parse_args()

if args.use_gpu:
    device = 'cuda'
else:
    device = 'cpu'

lora_alpha = args.lora_alpha
base_model = args.base_model
lora = args.lora
output = args.output


# if args.layer_filter:
#     layers = []
#     for layer in args.layer_filter.split(' '):
#         if layer.isdecimal():
#             layers.append(int(layer))
#         elif '-' in layer:
#             start,_,end = layer.partition('-')
#             start,end = int(start),int(end)
#             layers.extend(range(start,end+1))
#         else:
#             raise NotImplementedError("layer_filter Not implemented:",args.layer_filter)
#     layers = sorted(set(layers))
#     layer_prefixes = tuple(f"blocks.{l}." for l in layers)
#     def filter_keys(keys):
#         new_keys = []
#         for key in keys:
#             if key.startswith("blocks."):
#                 if not key.startswith(layer_prefixes):
#                     continue
#             new_keys.append(key)
#         return new_keys

# else:
#     def filter_keys(keys):
#         return keys

def get_filter_keys_and_merge_coef(layer_filter):
    if layer_filter:
        layers = []
        layer_coef = {}
        for layer in layer_filter.split(' '):
            if '*' in layer:
                coef,_,layer = layer.partition('*')
                coef = float(coef)
            else:
                coef = 1 
            if layer.isdecimal():
                layers.append(int(layer))
                layer_coef[int(layer)]=coef
            elif '-' in layer:
                start,_,end = layer.partition('-')
                start,end = int(start),int(end)
                layers.extend(range(start,end+1))
                for l in range(start,end+1):
                    layer_coef[l] = coef
            else:
                raise NotImplementedError("layer_filter Not implemented:",layer_filter)
        layers = sorted(set(layers))
        layer_prefixes = tuple(f"blocks.{l}." for l in layers)
        def filter_keys(keys): 
            new_keys = []
            for key in keys:
                if key.startswith("blocks."): #过滤掉blocks开头，且不在允许范围内的权重
                    if not key.startswith(layer_prefixes):
                        continue
                new_keys.append(key)
            return new_keys
        def merge_coef(key):
            if key.startswith('blocks.') and int(key.split('.')[1]) in layer_coef:
                return layer_coef[int(key.split('.')[1])]
            else:
                return 1
    else:
        def filter_keys(keys):
            return keys
        def merge_coef(key):
            return 1
    return filter_keys,merge_coef

filter_keys,merge_coef = get_filter_keys_and_merge_coef(args.layer_filter)

with torch.no_grad():
    w: Dict[str, torch.Tensor] = torch.load(base_model, map_location='cpu')
    # merge LoRA-only slim checkpoint into the main weights
    w_lora: Dict[str, torch.Tensor] = torch.load(lora, map_location='cpu')
    # pdb.set_trace() #DEBUG
    for k in filter_keys(w_lora.keys()): #处理time_mixing之类的融合
        w[k] = w_lora[k]
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
                print(f'merging {lora_A} and {lora_B} into {k}')
                assert w[lora_B].shape[1] == w[lora_A].shape[0]
                lora_r = w[lora_B].shape[1]
                w[k] = w[k].to(device=device)
                w[lora_A] = w[lora_A].to(device=device)
                w[lora_B] = w[lora_B].to(device=device)
                w[k] += w[lora_B] @ w[lora_A] * (lora_alpha / lora_r) * merge_coef(k)
                output_w[k] = w[k].to(device='cpu', copy=True)
                del w[k]
                del w[lora_A]
                del w[lora_B]
                continue

        if 'lora' not in k:
            print(f'retaining {k}')
            output_w[k] = w[k].clone()
            del w[k]

    torch.save(output_w, output)
