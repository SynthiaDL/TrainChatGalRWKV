########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import json, math, random, os, sys,re
import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch_lightning.utilities import rank_zero_info
from .binidx import MMapIndexedDataset
from .utils import MaybeIsPrime

from transformers import PreTrainedTokenizerFast
from src.rwkv_tokenizer import TRIE_TOKENIZER

if os.environ.get("RWKV_VOCAB") == '20B':
    TOKEN_NEWLINE = 187
    # DOUBLE_NEWLINE = torch.tensor([187,187])
    # DOUBLE_NEWLINE = [187,187]
    # ALICE_RESPONSE_PREFIX = torch.tensor([187,187,2422,547,27])
    class WrappedTokenizer():
        def __init__(self,tokenizer_file="../20B_tokenizer") -> None:
            self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
        def encode(self,src,return_offsets_mapping=False,max_length=None):
            tokenized = self.tokenizer(src,return_offsets_mapping=return_offsets_mapping,max_length=max_length,truncation=max_length is not None)
            if return_offsets_mapping:
                return tokenized.input_ids, tokenized.offset_mapping
            else:
                return tokenized.input_ids
else:
    TOKEN_NEWLINE = 11
    class WrappedTokenizer():
        def __init__(self,tokenizer_file="../rwkv_vocab_v20230424.txt") -> None:
            self.tokenizer = TRIE_TOKENIZER(tokenizer_file)
        def encode(self,src,return_offsets_mapping=False,max_length=None):
            return self.tokenizer.encode(src,return_offsets=return_offsets_mapping,max_length=max_length)

INSTRUCT_PROMPTS = [
    "> {user}正在和{bot}交流，{bot}非常聪明，正在帮助{user}解决问题。",
    "> {bot}是一个知识渊博的专家，正在和{user}交谈。",
    "> {bot}懂得数学、物理等各种学科，经常帮助{user}。",
]
class MyDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.ctx_len = args.initial_ctx_len or args.ctx_len
        self.final_ctx_len = args.ctx_len
        self.warm_up_steps = max(args.ctx_warmup_steps*args.micro_bsz * (args.accumulate_grad_batches or 1),1)
        self.current_warmup_step = 0
        self.update_per_steps = args.micro_bsz * (args.accumulate_grad_batches or 1)
        
        if args.data_type == "binidx":
            self.vocab_size = args.vocab_size
            rank_zero_info(f"Current vocab size = {self.vocab_size} (make sure it's correct)")

            if args.data_file.endswith('/'):
                d_all = []
                for p in os.listdir(args.data_file):
                    if p.endswith(".idx"):
                        d_all += [p[:-4]]
                d_all.sort()
                rank_zero_info(d_all)
                exit(0)
            else:
                self.data = MMapIndexedDataset(args.data_file)
                self.data_size = len(self.data._bin_buffer) // self.data._index._dtype_size
                rank_zero_info(f"Data has {self.data_size} tokens.")

            if args.instruct_data_file:
                self.main_data = self.data
                self.main_data_size = self.data_size
                # self.instruct_data = MMapIndexedDataset(args.instruct_data_file)
                # self.instruct_data_size = len(self.instruct_data._bin_buffer) // 2
                # rank_zero_info(f"Extra instruct chat Data has {self.instruct_data_size} tokens.")
                self.instruct_data = []
                with open(args.instruct_data_file) as f:
                    for line in f:
                        d = json.loads(line)
                        self.instruct_data.append(d['text'])
                #硬编码！！！
                # 读取names,然后替换Alice Bob人名
                if args.names_data_file:
                    with open(args.names_data_file) as f:
                        self.names_data = json.load(f)
                        self.max_replace = 10
                self.tokenizer = WrappedTokenizer()
                

            if args.my_qa_mask > 0:
                self.data_pile = MMapIndexedDataset('/fsx/BlinkDL/pile/pile_20B_tokenizer_text_document')
                self.data_pile_size = len(self.data_pile._bin_buffer) // self.data_pile._index._dtype_size

            if args.my_pile_stage > 0:
                # assert self.data_size == 332115325534 and self.vocab_size == 50277
                self.samples_per_epoch = args.epoch_steps * args.real_bsz
                assert self.samples_per_epoch == 40320
                rank_zero_info(f"########## Pile 20b-tokenized stage {args.my_pile_stage} ##########")
                dataset_slot = self.data_size // self.ctx_len
                if args.my_pile_stage != 4:
                    assert MaybeIsPrime(args.magic_prime)
                    assert args.magic_prime % 3 == 2
                    assert args.magic_prime / dataset_slot > 0.99 and args.magic_prime / dataset_slot <= 1
        elif args.data_type == "numpy":
            self.data = np.load(args.data_file).astype("int")
            self.vocab_size = args.vocab_size
            rank_zero_info("Current vocab size =", self.vocab_size, "(make sure it's correct)")
            self.data_size = len(self.data)
            rank_zero_info(f"Data has {self.data_size} tokens.")
        elif args.data_type == "uint16":
            self.data = np.fromfile(args.data_file, dtype=np.uint16).astype("int32").reshape(-1, args.my_sample_len)
            self.vocab_size = args.vocab_size
            rank_zero_info("Current vocab size =", self.vocab_size, "(make sure it's correct)")
            self.data_size = self.data.shape[0]
            rank_zero_info(f"Data has {self.data_size} samples.")
        elif args.data_type == "wds_img":
            self.vocab_size = -1
            self.data_size = -1
            self.data = None
            self.error_count = 0
        else:
            if args.data_type == "dummy":
                rank_zero_info("Building dummy data...")
                self.data = ""
                for i in range(100000):
                    aa = (i) % 10000
                    bb = (i * i) % 10000
                    cc = aa + bb
                    self.data += f".{aa}+{bb}={cc}."
            else:
                self.data = open(args.data_file, "r", encoding=args.data_type).read()
            rank_zero_info("Building token list...")
            unique = sorted(list(set(self.data)))
            self.vocab_size = len(unique)
            # rank_zero_info()
            # for u in unique:
            #     print(u, end=' ')
            # rank_zero_info('\n\n')
            xx = 0
            xxObj = {}
            for u in unique:
                xxObj[xx] = u
                xx += 1
            with open(f"{args.proj_dir}/vocab.json", "w", encoding="utf-16le") as vocab_file:
                vocab_file.write(json.dumps(xxObj, ensure_ascii=False))
            self.data_size = len(self.data)
            rank_zero_info(f"Data has {self.data_size} tokens, {self.vocab_size} vocab size.")
            self.stoi = {ch: i for i, ch in enumerate(unique)}
            self.itos = {i: ch for i, ch in enumerate(unique)}

    def sample_name_pair(self):
        if self.args.names_data_file:
            doc_names = {}
            while len(doc_names)<2:
                doc_names, = random.choices(self.names_data['doc_names'],weights = self.names_data['doc_weights'])
                names,name_weights = list(doc_names.keys()),list(doc_names.values())
            i1, = random.choices(list(range(len(names))),weights = name_weights)
            name_weights[i1]=0
            i2, = random.choices(list(range(len(names))),weights = name_weights)
            return names[i1],names[i2]
        else:
            return "Bob", "Alice"
    

    def find_someone_dialogue_span(self,text,name, 
        # pattern=r"(?:^|\n\n)(?P<name>{name})\:(?P<content>.*?)(?=\n\n|$)"):
        pattern=r"(?:^|(?<=\n\n))(?P<name>{name})\:(?P<content>.*?(\n\n|$))"):
        pattern = pattern.format(name=re.escape(name))
        # pattern=r"(?:^|\n\n)(?P<name>\w+)\:(?P<content>.*?)(?=\n\n|$)"):
        return [m.span("content") for m in re.finditer(pattern,text,re.DOTALL)]

    def tokenize_dialogue(self,text,target_name=None,max_length=None):
        input_ids,offset_mapping = self.tokenizer.encode(text,return_offsets_mapping=True,max_length=max_length)
        dix = input_ids
        # x = dix[:-1]
        # y = dix[1:]
        # z = [1] * len(x)
        dix_with_mask = dix.copy()
        if target_name:
            name_spans = self.find_someone_dialogue_span(text,target_name)
            if name_spans:
                span_start,span_end = name_spans.pop(0)
                for i,(start,end) in enumerate(offset_mapping):
                    if end <= span_start or start>=span_end:
                        dix_with_mask[i] = -100
                        # z[i] = 0
                    if start>=span_end and name_spans:
                        span_start,span_end = name_spans.pop(0)
            else:
                print("Cannot find name spans", target_name)
                print(text)
                dix_with_mask = [-100] * len(dix)
        return dix,dix_with_mask
    def __len__(self):
        return self.args.epoch_steps * self.args.micro_bsz

    def __getitem__(self, idx):
        args = self.args
        rank = self.global_rank
        epoch = self.real_epoch
        world_size = self.world_size
        if self.ctx_len<self.final_ctx_len:
            if self.current_warmup_step%self.update_per_steps == 0:
                self.ctx_len = int(min(self.current_warmup_step/self.warm_up_steps,1)*\
                    (args.ctx_len-args.initial_ctx_len)+args.initial_ctx_len)
            self.current_warmup_step += 1
        ctx_len = self.ctx_len
        # if args.chat_enhance_data:
        #     p = random.random()
        #     if p < args.chat_enhance_ratio:
        #         pass #todo add belle chat data
        # print(f"epoch {epoch} idx {idx} rank {rank}/{world_size}")
        if args.instruct_data_file:
            #先只考虑batchsize=1！不考虑padding问题
            #之后或许可以做一个状态重置输入，这样就可以batch，不过可能得改cuda kernal了。
            if random.random()<args.instruct_data_ratio:
                dix_concat = []
                dix_with_mask_concat = []
                # y_concat = []
                # z_concat = []
                replaced_count = 0
                names = self.sample_name_pair()
                while len(dix_concat)<=ctx_len:
                    seq = random.choice(self.instruct_data).rstrip('\n')+'\n\n'
                    if replaced_count == 0:
                        seq = random.choice(INSTRUCT_PROMPTS).format(user=names[0],bot=names[1])+"\n\n" + seq
                    (seq,num_replaced) = re.subn(r"(^|\n)Bob: ",r"\g<1>"+names[0]+": ",seq) 
                    (seq,num_replaced) = re.subn(r"(^|\n)Alice: ",r"\g<1>"+names[1]+": ",seq) 
                    replaced_count += num_replaced
                    seq = seq.strip()
                    # dix,dix_with_mask = self.mask_except_for_name(seq,names[1])
                    dix,dix_with_mask = self.tokenize_dialogue(seq,names[1])
                    if replaced_count >= self.max_replace:
                        replaced_count = 0
                        names = self.sample_name_pair()
                    # dix += DOUBLE_NEWLINE
                    # if dix_with_mask[-1] == -100:
                    #     dix_with_mask += [-100,-100]
                    # else:
                    #     dix_with_mask += DOUBLE_NEWLINE

                    
                    # dix = self.tokenizer(seq).input_ids
                    # dix = dix + DOUBLE_NEWLINE
                    # x = dix[:-1].copy()
                    # y = dix[1:].copy()
                    # z = [1] * len(x)
                    # count = 0
                    # for i in range(len(DOUBLE_NEWLINE),len(dix)):
                    #     if (dix[i-2:i] == DOUBLE_NEWLINE):
                    #         count+=1
                    #     if count>=2:
                    #         for j in range(i):
                    #             y[j] = -100
                    #             z[j] = 0
                    #         break
                    dix_concat += dix
                    dix_with_mask_concat += dix_with_mask
                x = torch.tensor(dix_concat[:ctx_len],dtype=torch.long)
                y = torch.tensor(dix_with_mask_concat[1:ctx_len+1],dtype=torch.long)
                z = torch.ones((ctx_len,),dtype=torch.bfloat16)
                z[y==-100] = 0.
                return (
                    x,y,z
                )

            # if random.random() < args.instruct_data_ratio:
            #     x_concat = torch.tensor([],dtype=torch.long)
            #     y_concat = torch.tensor([],dtype=torch.long)
            #     z_concat = torch.tensor([],dtype=torch.long)
            #     while len(x_concat)<ctx_len:
            #         dix = random.choice(self.instruct_data)
            #         dix = torch.tensor(dix.astype(int))
            #         x = torch.zeros(len(dix)+1,dtype=torch.long) + TOKEN_NEWLINE
            #         x[:len(dix)-1] = dix[:-1]
            #         y = torch.zeros(len(dix)+1,dtype=torch.long) + TOKEN_NEWLINE
            #         y[:len(dix)-1] = dix[1:]
            #         z = torch.ones(len(dix)+1, dtype=torch.bfloat16)
            #         for i in range(len(ALICE_RESPONSE_PREFIX),len(dix)):
            #             if (dix[i-5:i] == ALICE_RESPONSE_PREFIX).all():
            #                 y[:i] = -100
            #                 z[:i] = 0
            #                 break
            #         x_concat = torch.cat((x_concat,x))
            #         y_concat = torch.cat((y_concat,y))
            #         z_concat = torch.cat((z_concat,z))
            #     return x_concat[:ctx_len].contiguous(), y_concat[:ctx_len].contiguous(), z_concat[:ctx_len].contiguous()

                    
                

        if args.data_type == "wds_img":
            def init_wds(self, bias=0):
                def identity(x):
                    return x            
                import webdataset as wds
                import torchvision.transforms as transforms
                # img_transform = transforms.Compose(
                #     [transforms.CenterCrop(256)]
                # )
                img_transform = transforms.Compose([
                    transforms.CenterCrop(512),
                    transforms.Resize((args.my_img_size))
                ])
                self.data_raw = wds.WebDataset(args.data_file, resampled=True).shuffle(10000, initial=1000, rng=random.Random(epoch*100000+rank+bias*1e9)).decode("torchrgb").to_tuple("jpg", "json", "txt").map_tuple(img_transform, identity, identity)
                for pp in self.data_raw.pipeline:
                    if 'Resampled' in str(pp):
                        pp.deterministic = True
                        def worker_seed():
                            return rank*100000+epoch+bias*1e9
                        pp.worker_seed = worker_seed
                self.data = iter(self.data_raw)
                # print(f"WebDataset loaded for rank {rank} epoch {epoch}")
            if self.data == None:
                init_wds(self)
            trial = 0
            while trial < 10:
                try:
                    dd = next(self.data) # jpg, json, txt
                    break
                except:
                    print(f'[dataloader error - epoch {epoch} rank {rank} - trying a new shuffle]')
                    self.error_count += 1
                    init_wds(self, self.error_count)
                    trial += 1
                    pass
            # print(f"epoch {epoch} idx {idx} rank {rank}/{world_size} {dd[2]}")
            # with open(f"sample_{rank}.txt", "a", encoding="utf-8") as tmp:
            #     tmp.write(f"epoch {epoch} idx {idx} rank {rank}/{world_size} {int(dd[1]['key'])}\n")
            return dd[0], dd[2]
        else:
            if args.data_type == "uint16":
                i = np.random.randint(0, self.data_size-1)
                dix = self.data[i]
                x = torch.tensor(dix[:-1], dtype=torch.long)
                y = torch.tensor(dix[1:], dtype=torch.long)
            else:
                req_len = ctx_len + 1
                magic_prime = args.magic_prime
                data = self.data

                if args.my_pile_stage > 0 and args.my_pile_stage != 4:
                    ii = 1 + epoch * self.samples_per_epoch + (idx * world_size) + rank

                    if args.my_qa_mask > 0:
                        ii_orig = ii
                        if ii % 2 == 0:
                            ii = (ii // 2) * args.magic_prime
                            if self.ctx_len == 1024:
                                magic_prime = 324331313
                            elif self.ctx_len == 2048:
                                magic_prime = 162165671
                            elif self.ctx_len == 4096:
                                magic_prime = 81082817
                            data = self.data_pile
                        else:
                            ii = ii // 2

                    factor = (math.sqrt(5) - 1) / 2
                    factor = int(magic_prime * factor)
                    i = ((factor * ii * ii * ii) % magic_prime) * ctx_len
                    if (args.my_qa_mask == 0) or (data == self.data_pile):
                        i = i + args.my_pile_shift
                    # print(f"epoch {epoch} idx {idx} rank {rank}/{world_size} ii {ii} pos {round(i / self.data_size, 3)}")
                else:
                    # cheat: pick a random spot in dataset
                    i = np.random.randint(0, self.data_size - req_len)
                if args.data_type == "binidx":
                    dix = data.get(idx=0, offset=i, length=req_len).astype(int)
                    if args.my_script_align == 1:
                        #实现功能：对齐脚本，保证开头是说话人或者旁白描述（即刚刚经历过双重换行）
                        #做法是在dix里找到第一处双重换行，然后重新从此处构造dix
                        raise NotImplementedError("因不支持world,脚本对齐已废弃")
                        count_newline = 0
                        j = 0
                        while True:
                            if i == 0:
                                break #如果是全局开头，现在的dix就可以用
                            if dix[j]==TOKEN_NEWLINE: #如果遇到换行，计数器加一
                                count_newline += 1
                            else: #如果不是换行符，则进行判断
                                if count_newline >= 2: #已经经历了两个以上的连续换行
                                    if i < self.data_size-req_len: #如果i+req_len不超限，直接构造新的dix
                                        dix = data.get(idx=0, offset=i, length=req_len).astype(int)
                                        break
                                    else: #如果i+req_len超限了，重新选择i,然后重新搜索
                                        i = np.random.randint(0, self.data_size - req_len)
                                        dix = data.get(idx=0, offset=i, length=req_len).astype(int)
                                        j = 0
                                        count_newline = 0
                                        continue
                                else: #当前不是换行，并且经历了不到2个换行，则重置计数器
                                    count_newline = 0
                            i += 1
                            j += 1
                            if j >= req_len: #j超出了dix索引范围，则重新选择i,然后重新搜索
                                i = np.random.randint(0, self.data_size - req_len)
                                dix = data.get(idx=0, offset=i, length=req_len).astype(int)
                                j = 0
                                count_newline = 0
                elif args.data_type == "numpy":
                    dix = data[i : i + req_len]
                else:
                    dix = [self.stoi[s] for s in data[i : i + req_len]]

                if args.my_qa_mask == 1:
                    if data == self.data_pile:
                        z = [1] * ctx_len
                    else:
                        z = [0] * ctx_len
                        z_sum = 0
                        isGood = False
                        for i in range(3, ctx_len):
                            if dix[i] == 27 and dix[i-1] == 34 and dix[i-2] == 187 and dix[i-3] == 187:
                                isGood = True
                            if dix[i] == 0:
                                isGood = False
                            if isGood:
                                z[i] = 1
                                z_sum += 1
                        if z_sum == 0:
                            z = [1] * ctx_len
                            i = np.random.randint(0, self.data_pile_size - req_len)
                            dix = self.data_pile.get(idx=0, offset=i, length=req_len).astype(int)
                    z = torch.tensor(z, dtype=torch.bfloat16)
                if args.my_script_mask:
                    #脚本训练目标mask
                    #是一个字典{“cut_extra”:True, "mask_prefix_lines":8}
                    #包括两部分1：mask掉跨越作品的多余部分
                    #2：mask掉开头的几行，但是这个不超越dix的一半，也不超过作品的边界
                    prefix_max_len = ctx_len//2
                    z = [1] * ctx_len
                    z = torch.tensor(z, dtype=torch.bfloat16) #Or Bool?
                    if args.my_script_mask.get('cut_extra'):
                        assert args.data_type == 'binidx', "Only support Binidx"
                        docid,docstart,doclength = self.data.inverse_index(i)
                        real_length = docstart+doclength-(i+1) #y从i+1处开始
                        if real_length < ctx_len:
                            z[real_length:] = 0
                        prefix_max_len = min(prefix_max_len,real_length)
                    if args.my_script_mask.get('mask_prefix_lines'):
                        raise NotImplementedError("因不支持World，mask前缀行已废弃")
                        mask_prefix_lines = args.my_script_mask['mask_prefix_lines']
                        newline_counter = 0
                        for j in range(prefix_max_len):
                            if dix[j] == TOKEN_NEWLINE:
                                newline_counter += 1
                            if newline_counter == mask_prefix_lines:
                                break
                        if newline_counter == mask_prefix_lines:
                            z[:j]=0
                    elif args.my_script_mask.get("mask_prefix_tokens"):
                        z[:args.my_script_mask['mask_prefix_tokens']] = 0
                x = torch.tensor(dix[:-1], dtype=torch.long)
                y = torch.tensor(dix[1:], dtype=torch.long)

                # if ii_orig < 50:
                #     # if rank == 1:
                #     print('rank', rank, 'i', ii_orig, ii, i, 'x', x[:5], '...', x[-5:])
                # else:
                #     exit(0)

                if args.my_qa_mask == 1 or args.my_script_mask:
                    y[z==0] = -100
                    return x, y, z

            return x, y


class WNDataset(Dataset):
    def __init__(self, args):
        self.args = args
        #WNDataset，只用来处理定向预测数据。
        #用于处理定向预测特定人物发言的数据，或者总结数据
        #---Global Info---
        #以下为一段发生在嘉祥和巧克力之间的互动片段。
        #嘉祥是一名男性人类。巧克力是一名女性猫娘。
        

    def __len__(self):
        return self.args.epoch_steps * self.args.micro_bsz

    def __getitem__(self, idx):
        args = self.args
        rank = self.global_rank
        epoch = self.real_epoch
        world_size = self.world_size
        

        
