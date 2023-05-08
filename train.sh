
# cd RWKV-v4neo/ ;\
# python train.py --load_model ../pretrained_models/rwkv-4-novel/RWKV-4-Novel-3B-v1-ChnEng-20230412-ctx4096.pth \
#   --wandb chatgal3b --data_file ../data/chatgal_text_document --data_type binidx \
#   --vocab_size 50277 --ctx_len 4096 --accumulate_grad_batches 4 --epoch_steps 1000 \
#   --epoch_count 20 --epoch_save 1 --micro_bsz 4 --n_layer 32 --n_embd 2560 \
#   --pre_ffn 0 --head_qk 0 --lr_init 1e-5 --lr_final 1e-5 --warmup_steps 50 \
#   --beta1 0.9 --beta2 0.999 --adam_eps 1e-8 --accelerator gpu --devices 4 \
#   --precision bf16 --strategy deepspeed_stage_2_offload --grad_cp 1


python train.py --load_model ../pretrained_models/rwkv-4-raven/RWKV-4-Raven-7B-v10x-Eng49%-Chn50%-Other1%-20230423-ctx4096.pth  \
  --proj_dir out7b --wandb chatgal7b --data_file ../data/chatgal_0424_colonspace_bracketaside_text_document --data_type binidx \
  --vocab_size 50277 --ctx_len 4096 --accumulate_grad_batches 8 --epoch_steps 1000 \
  --epoch_count 20 --epoch_save 1 --epoch_begin 0 --micro_bsz 2 --n_layer 32 --n_embd 4096 \
  --pre_ffn 0 --head_qk 0 --lr_init 5e-6 --lr_final 1e-6 --warmup_steps 50 \
  --beta1 0.9 --beta2 0.999 --adam_eps 1e-8 --accelerator gpu --devices 4 \
  --precision bf16 --strategy deepspeed_stage_2_offload --grad_cp 1


# cd RWKV-v4neo-lora/ 
# python train.py \
#   --load_model ../pretrained_models/rwkv-4-raven/RWKV-4-Raven-7B-v10x-Eng49%-Chn50%-Other1%-20230423-ctx4096.pth  \
#   --proj_dir out7blora --wandb chatgal7blora\
#   --data_file ../data/ \
#   --data_type binidx \
#   --vocab_size 50277 --ctx_len 4096 --epoch_steps 1000 --epoch_count 20 \
#   --epoch_begin 0 --epoch_save 1 --micro_bsz 32 --n_layer 32 --n_embd 4096 \
#   --pre_ffn 0 --head_qk 0 --lr_init 1e-5 --lr_final 1e-5 --warmup_steps 1000 \
#   --beta1 0.9 --beta2 0.999 --adam_eps 1e-8 --accelerator gpu --devices 1 \
#   --precision bf16 --grad_cp 1 --accumulate_grad_batches 1 --strategy deepspeed \
#   --my_script_align 1 --my_script_mask 1_4 \
#   --lora --lora_r 16 --lora_alpha 32 --lora_dropout 0.01 \
#   --lora_parts=att,ffn,time,ln # --lora_load <lora checkpoint to continue training>

  #lora运行有很多玄学问题
  #首先lightning的单卡支持越来越废物了
  #fsdp似乎lora还不支持
  #然后如果不开梯度检查点，莫名其妙会爆显存
  #姑且就这样吧，用最基础的deepspeed策略


python train.py \
  --load_model ../pretrained_models/rwkv-4-raven/RWKV-4-Raven-7B-v10x-Eng49%-Chn50%-Other1%-20230423-ctx4096.pth  \
  --proj_dir out7b_lora-0426_r16_b32_c4096_1e-5 --wandb chatgal7blora\
  --data_file ../data/chatgal_0426_myclean_text_document \
  --data_type binidx \
  --vocab_size 50277 --ctx_len 4096 --epoch_steps 1000 --epoch_count 20 \
  --epoch_begin 0 --epoch_save 1 --micro_bsz 1 --n_layer 32 --n_embd 4096 \
  --pre_ffn 0 --head_qk 0 --lr_init 1e-5 --lr_final 5e-5 --warmup_steps 1000 \
  --beta1 0.9 --beta2 0.999 --adam_eps 1e-8 --accelerator gpu --devices 1 \
  --precision bf16 --grad_cp 1 --accumulate_grad_batches 1 --strategy deepspeed \
  --my_script_align 1 --my_script_mask 1_4 \
  --lora --lora_r 16 --lora_alpha 32 --lora_dropout 0.01 \
  --lora_parts=att,ffn,time,ln --lora_layers "25-31"
