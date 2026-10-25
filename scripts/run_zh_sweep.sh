#!/bin/bash
cd /home/shaokaolanmei/ai/minimind/trainer || exit 1
export PYTORCH_ALLOC_CONF=expandable_segments:True
for r in 8 16 32; do
  echo "=== rank $r ==="
  /home/shaokaolanmei/ai/minimind-venv/bin/python -u train_lora.py \
    --lora_rank $r --lora_name zh_r$r --from_weight full_sft_sub \
    --data_path ../dataset/lora_identity_zh.jsonl \
    --epochs 2 --batch_size 4 --max_seq_len 340 --learning_rate 1e-4 \
    --log_interval 20 --save_interval 200 --num_workers 0 2>&1 | tee ../logs/zh_r$r.log
  echo "=== done rank $r ==="
done
echo ALLDONE