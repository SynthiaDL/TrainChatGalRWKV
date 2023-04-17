#超参数随便填的
cd RWKV-v4neo/ ;\
python train.py --load_model ../pretrained_models/rwkv-4-novel/RWKV-4-Novel-3B-v1-ChnEng-20230412-ctx4096.pth \
  --wandb chatgal3b --data_file ../data/chatgal_text_document --data_type binidx \
  --vocab_size 50277 --ctx_len 4096 --accumulate_grad_batches 4 --epoch_steps 1000 \
  --epoch_count 20 --epoch_save 1 --micro_bsz 4 --n_layer 32 --n_embd 2560 \
  --pre_ffn 0 --head_qk 0 --lr_init 1e-5 --lr_final 1e-5 --warmup_steps 50 \
  --beta1 0.9 --beta2 0.999 --adam_eps 1e-8 --accelerator gpu --devices 4 \
  --precision bf16 --strategy deepspeed_stage_2_offload --grad_cp 1