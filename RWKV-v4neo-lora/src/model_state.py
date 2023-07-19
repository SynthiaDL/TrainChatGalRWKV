########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import functools
import os, math, gc, importlib
from typing import List, Optional
import numpy as np
import torch
import pdb
# torch._C._jit_set_profiling_executor(True)
# torch._C._jit_set_profiling_mode(True)
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from pytorch_lightning.strategies import DeepSpeedStrategy
if importlib.util.find_spec('deepspeed'):
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

# from deepspeed.runtime.fp16.onebit.zoadam import ZeroOneAdam

LORA_CONFIG = {
    "r": 0,
    "alpha": 0,
    "dropout": 0,
    "parts": {"att", "ln", "time"},
    "layers":None,
}

global_args = None

# try:
#     print('RWKV_MY_TESTING', os.environ["RWKV_MY_TESTING"])
# except:
#     os.environ["RWKV_MY_TESTING"] = ''

def __nop(ob):
    return ob


MyModule = nn.Module
MyFunction = __nop
if os.environ["RWKV_JIT_ON"] == "1":
    MyModule = torch.jit.ScriptModule
    MyFunction = torch.jit.script_method

class TimeMixState:

    def __init__(self, shift_state: torch.Tensor, wkv_state: torch.Tensor):
        self.shift_state = shift_state
        self.wkv_state = wkv_state


class ChannelMixState:

    def __init__(self, shift_state: torch.Tensor):
        self.shift_state = shift_state


class BlockState:

    def __init__(self, time_mix_state: TimeMixState,
                 channel_mix_state: ChannelMixState):
        self.time_mix_state = time_mix_state
        self.channel_mix_state = channel_mix_state


class BlockStateList:

    def __init__(self, shift_states, wkv_states):
        self.wkv_states = wkv_states
        self.shift_states = shift_states

    @staticmethod
    def create(N, B, C, device, dtype):
        result = BlockStateList.empty(N, B, C, device, dtype)
        result.wkv_states[:] = 0
        result.wkv_states[:, :, :, -1] = -1e38
        result.shift_states[:] = 0
        return result

    @staticmethod
    def empty(N, B, C, device, dtype):
        wkv_states = torch.empty((N, B, C, 3),
                                 device=device,
                                 dtype=torch.float)
        shift_states = torch.empty((N, 2, B, C), device=device, dtype=dtype)
        return BlockStateList(shift_states, wkv_states)

    def __getitem__(self, layer: int):
        return BlockState(
            TimeMixState(self.shift_states[layer, 0], self.wkv_states[layer]),
            ChannelMixState(self.shift_states[layer, 1]))

    def __setitem__(self, layer: int, state: BlockState):
        self.shift_states[layer, 0] = state.time_mix_state.shift_state
        self.wkv_states[layer] = state.time_mix_state.wkv_state
        self.shift_states[layer, 1] = state.channel_mix_state.shift_state


########################################################################################################
# CUDA Kernel
########################################################################################################

T_MAX = int(os.environ["RWKV_T_MAX"])  # TAKES LOTS OF VRAM!
# it's possible to go beyond CUDA limitations if you slice the ctx and pass the hidden state in each slice

from torch.utils.cpp_extension import load

if os.environ["RWKV_FLOAT_MODE"] == "bf16":
    wkv_cuda = load(name=f"wkv_{T_MAX}_bf16", sources=["cuda/wkv_op_state_bf16.cpp", "cuda/wkv_cuda_state_bf16.cu"], verbose=True, extra_cuda_cflags=["-t 4", "-std=c++17", "-res-usage", "--maxrregcount 60", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-DTmax={T_MAX}"])
    class WKV(torch.autograd.Function):
        # @staticmethod
        # def init_state(B, C):
        #     state = torch.zeros((B, C, 3), device='cuda')
        #     state[:,:,2] = -1e38
        #     return state.cuda()
        @staticmethod
        def forward(ctx, B, T, C, w, u, k, v,last_state):
            # global_args.forward_wkv_count += 1
            ctx.B = B
            ctx.T = T
            ctx.C = C
            assert T <= T_MAX
            assert B * C % min(C, 32) == 0
            w = -torch.exp(w.float().contiguous())
            u = u.contiguous()
            k = k.contiguous()
            v = v.contiguous()
            last_state = last_state.contiguous()
            y = torch.empty((B, T, C), device=w.device, memory_format=torch.contiguous_format, dtype=torch.bfloat16)
            new_state = torch.empty((B, C, 3), device=w.device, memory_format=torch.contiguous_format,dtype=torch.float32)
            wkv_cuda.forward(B, T, C, w, u, k, v, last_state, y,new_state)
            ctx.save_for_backward(w, u, k, v, y,last_state)
            return y,new_state

        @staticmethod
        def backward(ctx, gy,gnew_state):
            # global_args.backward_wkv_count += 1
            B = ctx.B
            T = ctx.T
            C = ctx.C
            assert T <= T_MAX
            assert B * C % min(C, 32) == 0
            w, u, k, v, y, last_state = ctx.saved_tensors
            gw = torch.empty((B, C), device=gy.device, memory_format=torch.contiguous_format, dtype=torch.bfloat16)
            gu = torch.empty((B, C), device=gy.device, memory_format=torch.contiguous_format, dtype=torch.bfloat16)
            gk = torch.empty((B, T, C), device=gy.device, memory_format=torch.contiguous_format, dtype=torch.bfloat16)
            gv = torch.empty((B, T, C), device=gy.device, memory_format=torch.contiguous_format, dtype=torch.bfloat16)
            glast_state = torch.empty((B, C, 3), device=w.device, memory_format=torch.contiguous_format,dtype=torch.float32)
            wkv_cuda.backward(B, T, C, w, u, k, v, last_state, y, gy.contiguous(), gnew_state.contiguous(), gw, gu, gk, gv, glast_state)
            gw = torch.sum(gw, dim=0)
            gu = torch.sum(gu, dim=0)
            return (None, None, None, gw, gu, gk, gv, glast_state)

else:
    wkv_cuda = load(name=f"wkv_{T_MAX}", sources=["cuda/wkv_op_state.cpp", "cuda/wkv_cuda_state.cu"], verbose=True, extra_cuda_cflags=["-res-usage", "--maxrregcount 60", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-DTmax={T_MAX}"])
    class WKV(torch.autograd.Function):
        @staticmethod
        def forward(ctx, B, T, C, w, u, k, v,last_state):
            ctx.B = B
            ctx.T = T
            ctx.C = C
            assert T <= T_MAX
            assert B * C % min(C, 32) == 0
            if "32" in os.environ["RWKV_FLOAT_MODE"]:
                w = -torch.exp(w.contiguous())
                u = u.contiguous()
                k = k.contiguous()
                v = v.contiguous()
            else:
                w = -torch.exp(w.float().contiguous())
                u = u.float().contiguous()
                k = k.float().contiguous()
                v = v.float().contiguous()
            last_state = last_state.contiguous()
            y = torch.empty((B, T, C), device=w.device, memory_format=torch.contiguous_format)
            new_state = torch.empty((B, C, 3), device=w.device, memory_format=torch.contiguous_format,dtype=torch.float32)
            wkv_cuda.forward(B, T, C, w, u, k, v, last_state, y,new_state)
            ctx.save_for_backward(w, u, k, v, y,last_state)
            if "32" in os.environ["RWKV_FLOAT_MODE"]:
                return y,new_state
            elif os.environ["RWKV_FLOAT_MODE"] == "fp16":
                return y.half(),new_state
            elif os.environ["RWKV_FLOAT_MODE"] == "bf16":
                return y.bfloat16(),new_state
        @staticmethod
        def backward(ctx, gy,gnew_state):
            B = ctx.B
            T = ctx.T
            C = ctx.C
            assert T <= T_MAX
            assert B * C % min(C, 32) == 0
            w, u, k, v, y, last_state = ctx.saved_tensors
            gw = torch.empty((B, C), device=gy.device, memory_format=torch.contiguous_format)
            gu = torch.empty((B, C), device=gy.device, memory_format=torch.contiguous_format)
            gk = torch.empty((B, T, C), device=gy.device, memory_format=torch.contiguous_format)
            gv = torch.empty((B, T, C), device=gy.device, memory_format=torch.contiguous_format)
            glast_state = torch.empty((B, C, 3), device=w.device, memory_format=torch.contiguous_format,dtype=torch.float32)
            if "32" in os.environ["RWKV_FLOAT_MODE"]:
                wkv_cuda.backward(B, T, C, w, u, k, v, y, gy.contiguous(), gnew_state.contiguous(), gw, gu, gk, gv, glast_state)
            else:
                wkv_cuda.backward(B, T, C, w, u, k, v, y, gy.float().contiguous(), gnew_state.contiguous(), gw, gu, gk, gv, glast_state)
            gw = torch.sum(gw, dim=0)
            gu = torch.sum(gu, dim=0)
            if "32" in os.environ["RWKV_FLOAT_MODE"]:
                return (None, None, None, gw, gu, gk, gv, glast_state)
            elif os.environ["RWKV_FLOAT_MODE"] == "fp16":
                return (None, None, None, gw.half(), gu.half(), gk.half(), gv.half(), glast_state)
            elif os.environ["RWKV_FLOAT_MODE"] == "bf16":
                return (None, None, None, gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16(), glast_state)

# if os.environ.get("RWKV_PARTS"): #内部切分片段，不暴露state接口。
#     num_parts = int(os.environ['RWKV_PARTS'])
#     def RUN_CUDA(B,T,C,w,u,k,v):
#         # assert(T%num_parts == 0)
#         t = (T+num_parts-1)//num_parts #向上取整，保证T中每个片段都被选取
#         assert(t<=T_MAX)
#         y_list = []
#         state = WKV_STATE.init_state(B, C)
#         for i in range(num_parts):
#             y, state = WKV_STATE.apply(B, t, C, w, u, k[:,t*i:t*(i+1),:], v[:,t*i:t*(i+1),:], state)
#             y_list.append(y)
#         return torch.cat(y_list, dim=1)
# else:
#     def RUN_CUDA(B, T, C, w, u, k, v):
#         return WKV.apply(B, T, C, w, u, k, v)
## wanicca's old
# def RUN_CUDA(B, T, C, w, u, k, v,state=None):
#     if state is None:
#         state = torch.zeros(
#             B,
#             C,
#             3,
#             dtype=torch.float32,
#             device=k.device,
#             memory_format=torch.contiguous_format,
#         )
#         state[:, :, 2] -= 1e38
#     return WKV.apply(B, T, C, w, u, k, v,state)

def RUN_CUDA(B, T, C, w, u, k, v, last_state):
    return WKV.apply(B, T, C, w, u, k, v, last_state)
########################################################################################################
# LoRA
########################################################################################################


class LoraLinear(nn.Module):

    def __init__(self, in_features: int, out_features: int, bias: bool):
        super().__init__()

        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        assert bias == False, "Biased LoraLinear not supported"

        r, alpha, dropout = LORA_CONFIG["r"], LORA_CONFIG[
            "alpha"], LORA_CONFIG["dropout"]
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, r))
        self.lora_dropout = nn.Dropout(dropout)
        self.scaling = alpha / r

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return (
            F.linear(x, self.weight) + self.scaling *
            F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B))


@functools.wraps(LoraLinear)
def make_linear_att(*args, **kwargs):
    if "att" in LORA_CONFIG["parts"] and LORA_CONFIG["r"] > 0:
        return LoraLinear(*args, **kwargs)
    else:
        return nn.Linear(*args, **kwargs)


@functools.wraps(LoraLinear)
def make_linear_ffn(*args, **kwargs):
    if "ffn" in LORA_CONFIG["parts"] and LORA_CONFIG["r"] > 0:
        return LoraLinear(*args, **kwargs)
    else:
        return nn.Linear(*args, **kwargs)


########################################################################################################
# RWKV: RWKV Time-mix + RWKV Channel-mix
########################################################################################################

#按照ChatRWKV的标准，state是一个向量数组，不考虑batch，i*5+(0=att_xx 1=att_aa 2=att_bb 3=att_pp 4=ffn_xx)
#按照HF的标准。state是一个张量数组，每个张量具有（Batch,Channel,Layer）的大小。state[0]是所有的ffn_xx, state[1]是所有的att_xx， 2,3,4分别对应aa,bb,pp
#为了方便起见，我们这里考虑state是一个二维数组，第一维度是layer，第二维度是state编号，存放的是(B,C)维度的状态张量
#而不是按huggingface的做法把所有层的状态放到一个张量，这样方便之后model parallel，不同层放到不同卡上。
#具体而言，对于每一层，state[0]是上一步的att_xx，state[1]是上一步的aa,bb,pp，state[2]是上一步的ffn_xx
#之后要是改huggingface，再考虑把aa,bb,pp这几个分开吧。现在这样减少拆分再拼接的次数。


class RWKV_TimeMix(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        # self.ctx_len = args.ctx_len
        self.ctx_len = T_MAX
        self.n_embd = args.n_embd

        with torch.no_grad():  # fancy init
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd

            # fancy time_decay
            decay_speed = torch.ones(args.dim_att)
            for h in range(args.dim_att):
                decay_speed[h] = -5 + 8 * (h / (args.dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed)
            # print(layer_id, self.time_decay.flatten()[:3].cpu().numpy(), '...', self.time_decay.flatten()[-3:].cpu().numpy())

            # fancy time_first
            zigzag = torch.tensor([(i + 1) % 3 - 1 for i in range(args.dim_att)]) * 0.5
            self.time_first = nn.Parameter(torch.ones(args.dim_att) * math.log(0.3) + zigzag)

            # fancy time_mix
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.time_mix_r = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        if LORA_CONFIG['layers'] is None or layer_id in LORA_CONFIG["layers"]:
            self.key = make_linear_att(args.n_embd, args.dim_att, bias=False)
            self.value = make_linear_att(args.n_embd, args.dim_att, bias=False)
            self.receptance = make_linear_att(args.n_embd, args.dim_att, bias=False)
        else:
            self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)
            self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
            self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)

        self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)

    @MyFunction
    def forward(self, x, last_state: TimeMixState):
        B, T, C = x.size()  # x = (Batch,Time,Channel)

        # Mix x with the previous timestep to produce xk, xv, xr
        xx = torch.concat((last_state.shift_state.unsqueeze(1), x[:, :-1]), dim=1)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)

        sr = torch.sigmoid(r)

        y, new_wkv_state = RUN_CUDA(B, T, C, self.time_decay, self.time_first,
                                    k, v, last_state.wkv_state)
        return self.output(sr * y), TimeMixState(x[:, -1], new_wkv_state)
    # @MyFunction
    # def jit_func(self, x, state):
    #     xx = self.time_shift(x) # Mix x with the previous timestep to produce xk, xv, xr
    #     if state is not None:
    #         xx[:, 0] = state
    #     xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
    #     xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
    #     xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
    #     k = self.key(xk)
    #     v = self.value(xv)
    #     r = self.receptance(xr)
    #     sr = torch.sigmoid(r)
    #     return sr, k, v, x[:,-1]

    # def forward(self, x, state):
    #     B, T, C = x.size()  # x = (Batch,Time,Channel)
    #     sr, k, v, x_ = self.jit_func(x,state)
    #     wkv, wkv_state = RUN_CUDA(B, T, self.args.dim_att, self.time_decay, self.time_first, k, v, state)
    #     rwkv = sr * wkv
    #     return self.output(rwkv), [x_,wkv_state]

    # if 'a' in os.environ["RWKV_MY_TESTING"]:
    #     @MyFunction
    #     def QKV(self, q, k, v):
    #         att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    #         att = att.masked_fill(self.att_mask == 0, float('-inf'))
    #         att = F.softmax(att, dim = -1)
    #         x = att @ v
    #         return x

    #     @MyFunction
    #     def jit_funcQKV(self, x):
    #         xx = self.time_shift(x) # Mix x with the previous timestep to produce xk, xv, xr
    #         xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
    #         xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
    #         xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
    #         xqq = x * self.time_mix_qq + xx * (1 - self.time_mix_qq)
    #         xkk = x * self.time_mix_kk + xx * (1 - self.time_mix_kk)
    #         xvv = x * self.time_mix_vv + xx * (1 - self.time_mix_vv)
    #         k = self.key(xk)
    #         v = self.value(xv)
    #         r = self.receptance(xr)
    #         sr = torch.sigmoid(r)
    #         qq = self.qq(xqq)
    #         kk = self.kk(xkk)
    #         vv = self.vv(xvv)
    #         return sr, k, v, qq, kk, vv

    #     def forward(self, x):
    #         B, T, C = x.size()  # x = (Batch,Time,Channel)
    #         sr, k, v, qq, kk, vv = self.jit_funcQKV(x)
    #         rwkv = sr * RUN_CUDA(B, T, self.args.dim_att, self.time_decay, self.time_first, k, v)
    #         rwkv = self.output(rwkv) + self.oo(self.QKV(qq, kk, vv))
    #         return rwkv

########################################################################################################

class RWKV_ChannelMix(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        # self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
        if LORA_CONFIG["layers"] is None or layer_id in LORA_CONFIG["layers"]:
            self.key = make_linear_ffn(args.n_embd, args.dim_ffn, bias=False)
            self.receptance = make_linear_ffn(args.n_embd, args.n_embd, bias=False)
            self.value = make_linear_ffn(args.dim_ffn, args.n_embd, bias=False)
        else:
            self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
            self.receptance = nn.Linear(args.n_embd, args.n_embd, bias=False)
            self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

    @MyFunction
    def forward(self, x, last_state: ChannelMixState):
        xx = torch.concat((last_state.shift_state.unsqueeze(1), x[:, :-1]), dim=1)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        k = self.key(xk)
        k = torch.square(torch.relu(k))
        kv = self.value(k)
        return torch.sigmoid(self.receptance(xr)) * kv, ChannelMixState(x[:, -1])

#     @MyFunction
#     def forward(self, x):
#         xx = self.time_shift(x)
#         xa = x * self.time_mix_k + xx * (1 - self.time_mix_k)
#         xb = x * self.time_mix_r + xx * (1 - self.time_mix_r)
#         a = self.aa(xa)
#         b = self.bb(xb)
#         return self.value(a * F.mish(b))

########################################################################################################
# The RWKV Model with our blocks
########################################################################################################


class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)
            # if args.my_pos_emb > 0:
            #     self.pos_emb_x = nn.Parameter(torch.zeros((1,args.my_pos_emb,args.n_embd)))
            #     self.pos_emb_y = nn.Parameter(torch.zeros((args.my_pos_emb,1,args.n_embd)))

        # if self.layer_id == 0 and self.args.pre_ffn > 0:
        #     self.ffnPre = RWKV_ChannelMix(args, 0)
        # else:
        #     self.att = RWKV_TimeMix(args, layer_id)
        self.att = RWKV_TimeMix(args, layer_id)

        self.ffn = RWKV_ChannelMix(args, layer_id)

        # if args.tiny_att_dim > 0 and self.layer_id == args.tiny_att_layer:
        #     self.tiny_ln = nn.LayerNorm(args.n_embd)
        #     self.tiny_q = nn.Linear(args.n_embd, args.tiny_att_dim, bias=False)
        #     self.tiny_k = nn.Linear(args.n_embd, args.tiny_att_dim, bias=False)
        #     self.tiny_v = nn.Linear(args.n_embd, args.n_embd, bias=False)
        #     # self.register_buffer("tiny_mask", torch.tril(torch.ones(args.ctx_len, args.ctx_len)))
        #     self.register_buffer("tiny_mask", torch.tril(torch.ones(T_MAX, T_MAX)))

    def forward(self, x, last_state: BlockState):
        args = self.args
        B, T, C = x.size()
        x_emb = x
        if self.layer_id == 0:
            x = self.ln0(x)

        att_out, att_state = self.att(
            self.ln1(x),
            last_state.time_mix_state,
        )
        x = x + att_out
        ffn_out, ffn_state = self.ffn(
            self.ln2(x),
            last_state.channel_mix_state,
        )
        x = x + ffn_out
        # if args.tiny_att_dim > 0 and self.layer_id == args.tiny_att_layer:
        #     xx = self.tiny_ln(x)
        #     q = self.tiny_q(xx)[:, :T, :]
        #     k = self.tiny_k(xx)[:, :T, :]
        #     c = (q @ k.transpose(-2, -1)) * (args.tiny_att_dim ** (-0.5))
        #     c = c.masked_fill(self.tiny_mask[:T, :T] == 0, 0)
        #     x = x + c @ self.tiny_v(x_emb)
        return x, BlockState(att_state, ffn_state)


class L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, y, token_amount):
        ctx.save_for_backward(y)
        ctx.token_amount = token_amount
        return loss

    @staticmethod
    def backward(ctx, grad_output): #这个函数会不会影响batch和grad_accu的一致性？感觉上会。梯度累积时，factor变大了。但是只有loss缩放，这里的正则化项反而没有缩放
        y = ctx.saved_tensors[0]
        # to encourage the logits to be close to 0
        if ctx.token_amount == 0:
            return (grad_output, None, None)
        factor = 1e-4 / ctx.token_amount #这一行类似crossentropy在token上平均。
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        if os.environ.get("WN_FIX_L2WRAP"): #实现batch等价性
            # maxx[maxx<3.]=0. #防止对已经较小的logits值下拉，只对大于阈值的往下拉
            gy.scatter_(-1, ids, maxx * factor * grad_output)
        else:
            gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy, None)
    #修改一下
    #正向 l2loss = L2(loss,y)
    #如果有梯度累积，那么dfinal/dl2loss=1/accumulate
    #所以应当在scatter_时加入grad_output
    #话说，这个名字为什么叫l2，明明也并不是2范数吧。


class RWKV(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        # global global_args
        # global_args= args
        # global_args.forward_wkv_count = 0
        # global_args.backward_wkv_count = 0
        self.args = args
        # if not hasattr(args, 'dim_att'):
        #     args.dim_att = args.n_embd
        # if not hasattr(args, 'dim_ffn'):
        #     args.dim_ffn = args.n_embd * 4
        # if not hasattr(args, 'tiny_att_layer'):
        #     args.tiny_att_layer = -1
        # if not hasattr(args, 'tiny_att_dim'):
        #     args.tiny_att_dim = -1

        self.emb = nn.Embedding(args.vocab_size, args.n_embd)

        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])

        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

        if args.head_qk > 0:
            self.head_q = nn.Linear(args.n_embd, args.head_qk, bias=False)
            self.head_k = nn.Linear(args.n_embd, args.head_qk, bias=False)
            # self.register_buffer("copy_mask", torch.tril(torch.ones(args.ctx_len, args.ctx_len)))
            self.register_buffer("copy_mask", torch.tril(torch.ones(T_MAX, T_MAX)))

    def configure_optimizers(self):
        args = self.args
        if args.layerwise_lr > 0:
            lr_1x = set()
            lr_2x = set()
            lr_3x = set()
            for n, p in self.named_parameters():
                if "time_mix" in n:
                    if args.my_pile_stage == 2:
                        lr_2x.add(n)
                    else:
                        lr_1x.add(n)
                elif "time_decay" in n:
                    if args.my_pile_stage == 2:
                        lr_3x.add(n)
                    else:
                        lr_2x.add(n)
                elif "time_first" in n:
                    lr_3x.add(n)
                else:
                    lr_1x.add(n)
            lr_1x = sorted(list(lr_1x))
            lr_2x = sorted(list(lr_2x))
            lr_3x = sorted(list(lr_3x))
            # print('1x', lr_1x)
            # print('2x', lr_2x)
            # print('3x', lr_3x)
            param_dict = {n: p for n, p in self.named_parameters()}
            no_decay = ['bias', 'LayerNorm.weight']
            if args.my_pile_stage == 2:
                optim_groups = [
                    {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0 if any(nd in n for nd in no_decay) else args.weight_decay, "my_lr_scale": 1.0},
                    {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0 if any(nd in n for nd in no_decay) else args.weight_decay, "my_lr_scale": 5.0},# test: 2e-3 / args.lr_init},
                    {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0 if any(nd in n for nd in no_decay) else args.weight_decay, "my_lr_scale": 5.0},# test: 3e-3 / args.lr_init},
                ]
            else:
                optim_groups = [
                    {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0 if any(nd in n for nd in no_decay) else args.weight_decay, "my_lr_scale": 1.0},
                    {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0 if any(nd in n for nd in no_decay) else args.weight_decay, "my_lr_scale": 2.0},
                    {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0 if any(nd in n for nd in no_decay) else args.weight_decay, "my_lr_scale": 3.0},
                ]
        else:
            optim_groups = [
                {"params": [p for n, p in self.named_parameters()], "weight_decay": 0.0 if any(nd in n for nd in no_decay) else args.weight_decay},
            ]

        for g in optim_groups:
            g["params"] = [p for p in g["params"] if p.requires_grad]
        optim_groups = [g for g in optim_groups if len(g["params"]) > 0]

        if self.deepspeed_offload:
            return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True)
        # return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adamw_mode=False, weight_decay=0, amsgrad=False)
        if self.args.strategy == 'single_device' and self.args.precision == 'bf16':
            return torch.optim.Adam(optim_groups,lr=self.args.lr_init,betas=self.args.betas, eps=self.args.adam_eps)
        else:
            return FusedAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True)
        # return ZeroOneAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, weight_decay=0, amsgrad=False, cuda_aware=False)

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return cfg.get("offload_optimizer") or cfg.get("offload_param")
        return False

    def forward(self, idx: torch.Tensor, last_shift_states: torch.Tensor,
                last_wkv_states: torch.Tensor):
        args = self.args
        B, T = idx.size()
        assert T <= T_MAX, "Cannot forward, model ctx_len is exhausted."

        x = self.emb(idx)
        new_states = BlockStateList.empty(args.n_layer, B, args.n_embd,
            x.device, x.dtype)
        for i,(block,block_state) in enumerate(zip(self.blocks,
            BlockStateList(last_shift_states, last_wkv_states))):
            # x = x.to(block.device)
            if args.grad_cp == 1 and i>0 : #and i < len(self.blocks)-1 
                if args.lora:
                    x, new_block_state = torch_checkpoint(block, x, block_state,use_reentrant=False) #use_reentrant=False或者不checkpoint第一层都可以使得梯度正常反传，要不然只反传一层
                else:
                    # x, new_block_state = deepspeed.checkpointing.checkpoint(block, x, block_state)
                    x, new_block_state = torch_checkpoint(block, x, block_state,use_reentrant=False) #use_reentrant=False或者不checkpoint第一层都可以使得梯度正常反传，要不然只反传一层
            else:
                x, new_block_state = block(x, block_state)
            new_states[i] = new_block_state
        # x = x.to(self.ln_out.device)

        x = self.ln_out(x)
        
        x = self.head(x)
        return x,new_states.shift_states, new_states.wkv_states
    
    def training_step(self, batch, batch_idx):
        args = self.args

        idx, targets, *others = batch
        B, T = idx.shape
        C = args.n_embd

        states = BlockStateList.create(args.n_layer, B, C, idx.device,
            self.emb.weight.dtype)
        # init_states = states
        # init_states.shift_states.requires_grad_()
        # init_states.wkv_states.requires_grad_()
        def checkpointed_step(idx, targets, prev_loss, last_shift_states,
                              last_wkv_states, prev_token_amount):
            logits, new_shift_states, new_wkv_states = self(idx, last_shift_states, last_wkv_states)
            current_token_amount = (targets!=-100).sum() #这样是不是更合适？
            # current_token_amount = idx.shape[1]
            if current_token_amount == 0:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1),reduction='sum')
            else:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
                loss = L2Wrap.apply(loss, logits, current_token_amount)
            new_token_amount = prev_token_amount+current_token_amount
            if new_token_amount>0:
                new_loss = prev_loss * (prev_token_amount / new_token_amount) + loss * (
                    current_token_amount / new_token_amount)
            else:
                new_loss = prev_loss
            return new_loss, new_shift_states, new_wkv_states, new_token_amount

        total_loss = torch.tensor(0.,dtype=self.emb.weight.dtype).requires_grad_()
        token_amount = 0
        # i = 0
        #Blealtan的做法是ctx_len定义为cuda核的大小（对应我们这里的T_max），然后引入ctx_len_cutoff作为控制状态重置的长度
        #然后T都是样本长度
        #我感觉类似ctx_len_cutoff以后还是用额外的输入来标记每个序列的重置点，而不是模型内部规定一个重置点。
        #所以这里就不改成Blealtan的思路了，不过稍后可以在他的基础上rebase。他的代码更简洁一些
        i = 0
        for i in range(math.ceil(T / T_MAX)-1):
            # pdb.set_trace()
            # total_loss, states, token_amount = deepspeed.checkpointing.checkpoint(
            total_loss,new_shift_states, new_wkv_states,token_amount = torch_checkpoint(
                checkpointed_step,
                idx[:, i * T_MAX:(i + 1) * T_MAX],
                targets[:, i * T_MAX:(i + 1) * T_MAX],
                total_loss,
                states.shift_states,
                states.wkv_states,
                token_amount,
                # use_reentrant=False
            )
            states = BlockStateList(new_shift_states, new_wkv_states)
            # if total_loss.isnan().all():
            #     import transformers
            #     tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_file="20B_tokenizer.json")
            #     pdb.set_trace()
        # pdb.set_trace()
        total_loss, new_shift_states, new_wkv_states, token_amount = checkpointed_step(
            idx[:, i * T_MAX:(i + 1) * T_MAX],
            targets[:, i * T_MAX:(i + 1) * T_MAX],
            total_loss,
            states.shift_states,
            states.wkv_states,
            token_amount
        )
        # pdb.set_trace()
        return total_loss
    
    # def training_step(self, batch, batch_idx):
    #     args = self.args
    #     if args.my_qa_mask != 1 and not args.my_script_mask:
    #         idx, targets = batch
    #         logits = self(idx)
    #         loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    #     else:
    #         idx, targets, mask = batch
    #         mask = mask.view(-1)
    #         sum_mask = torch.sum(mask).item()
    #         # if sum_mask == 0:
    #         #     return torch.tensor([0.0], requires_grad=True)
    #         logits = self(idx)
    #         if sum_mask == mask.shape[0]:
    #             loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    #             # print('rank', self.global_rank, 'loss', loss.item())
    #         else:
    #             loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none')
    #             # loss_raw = loss
    #             loss = torch.sum(loss * mask) / sum_mask
    #             # torch.set_printoptions(threshold=10000)
    #             # if True: #self.global_rank == 1:
    #             #     tmp = ''
    #             #     sss = 0
    #             #     ccc = 0
    #             #     for i in range(mask.shape[0]):
    #             #         if mask[i] > 0:
    #             #             tmp += str(idx.view(-1)[i].item()) + ','
    #             #             sss += loss_raw.view(-1)[i].float().item()
    #             #             ccc += 1
    #             #     print('rank', self.global_rank, 'loss', loss.item(), 'lavg', sss / ccc)#, 'tmp', tmp, 'input', idx)

    #     return L2Wrap.apply(loss, logits)

    def training_step_end(self, batch_parts):
        all_loss = self.all_gather(batch_parts)
        if self.trainer.is_global_zero:
            # self.trainer.my_loss_all = all
            with torch.no_grad():
                self.trainer.my_backward_step += 1
                self.trainer.my_loss += all_loss.float().mean().item()/self.trainer.accumulate_grad_batches
    @rank_zero_only
    def validation_step(self, batch, batch_idx):
        pass
        # seq = batch['input_ids']
        # assert isinstance(seq, torch.Tensor) and seq.ndim == 1
        # T, = idx.shape
        # C = self.n_embd
        # state = BlockStateList.create(self.n_layer, B, C, idx.device,
        #     self.emb.weight.dtype)
        # idx, target = seq[:-1], seq[1:]
        # loss = np.array([], dtype=np.float32)
        # for i in range(math.ceil(T / self.ctx_len)):
        #     logit, state = self(
        #         idx[i * self.ctx_len:(i + 1) * self.ctx_len].view(1, -1),
        #         state)
        #     piece_loss: np.ndarray = F.cross_entropy(
        #         logit,
        #         target[i * self.ctx_len:(i + 1) * self.ctx_len],
        #         reduction='none').float().cpu().numpy()
        #     loss = np.concatenate((loss, piece_loss))

        # print("validation loss shape: ", loss.shape)
        # exp_mean_loss = []
        # for i in range(8, math.ceil(math.log2(loss.shape[0]))):
        #     exp_mean_loss.append([i, loss[:min(len(loss), 2**i)].mean()])

        # print(exp_mean_loss)

        # import wandb
        # table = wandb.Table(data=exp_mean_loss,
        #                     columns=["length", "cross_entropy_loss"])
        # wandb.log({
        #     f"validation/loss_curve/{self.real_epoch}/{batch_idx}":
        #     wandb.plot.line(table,
        #                     "length",
        #                     "cross_entropy_loss",
        #                     title="Loss Curve"),
        # })

    def generate_init_weight(self):
        print(
            f"""
############################################################################
#
# Init model weight (slow for large models)...
#
############################################################################
"""
        )
        m = {}
        for n in self.state_dict():
            p = self.state_dict()[n]
            shape = p.shape

            gain = 1.0
            scale = 1.0
            if "ln_" in n or ".ln" in n or "time_" in n or "_mask" in n or "pos_emb" in n or '.mask.' in n:
                m[n] = p
            else:
                if n == "emb.weight":
                    scale = -1 * self.args.lr_init
                else:
                    if shape[0] > shape[1]:
                        gain = math.sqrt(shape[0] / shape[1])
                    for kk in [".att.key.", ".att.receptance.", ".att.output.", ".att.key.", ".ffn.value.", ".ffn.receptance.", ".ffnPre.value.", ".ffnPre.receptance.", "head_q.", '.oo.', '.rr.']:
                        if kk in n:
                            scale = 0
                    if n == "head.weight":
                        scale = 0.5
                    if "head_k." in n:
                        scale = 0.1
                    if "head_q." in n:
                        scale = 0

                print(f"{str(shape[0]).ljust(5)} {str(shape[1]).ljust(5)} {str(scale).ljust(4)} {n}")

                if self.args.accelerator.upper() == "GPU":
                    m[n] = torch.empty((shape[0], shape[1]), device="cuda")
                else:
                    m[n] = torch.empty((shape[0], shape[1]))

                if scale == 0:
                    nn.init.zeros_(m[n])
                elif scale < 0:
                    nn.init.uniform_(m[n], a=scale, b=-scale)
                else:
                    nn.init.orthogonal_(m[n], gain=gain * scale)

            m[n] = m[n].cpu()
            if os.environ["RWKV_FLOAT_MODE"] == "fp16":
                m[n] = m[n].half()
            elif os.environ["RWKV_FLOAT_MODE"] == "bf16":
                m[n] = m[n].bfloat16()

            # if n == "emb.weight":
            #     print(m[n])

        gc.collect()
        torch.cuda.empty_cache()
        return m
