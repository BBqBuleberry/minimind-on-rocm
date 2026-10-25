#!/bin/bash
cd /home/shaokaolanmei/ai/minimind || exit 1
for r in 8 16 32; do
  echo "########## rank $r ##########"
  printf '1\nhi\n你是谁？\n推荐一些中国美食\n' | /home/shaokaolanmei/ai/minimind-venv/bin/python -u eval_llm.py \
    --weight full_sft_sub --lora_weight zh_r$r --lora_rank $r \
    --max_new_tokens 96 --temperature 0.6 2>&1 | grep -v "Warning:" 
  echo
done
echo EVALDONE