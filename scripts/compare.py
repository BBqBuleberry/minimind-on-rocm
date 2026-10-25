import csv
rows = list(csv.DictReader(open("experiments.csv", encoding="utf-8")))
rows.sort(key=lambda r: int(r["rank"]))
print("| 实验 | rank | LoRA(M) | 首个loss | 末段loss | s/step | 状态 |")
print("|---|---|---|---|---|---|---|")
for r in rows:
    print(f"| {r['tag']} | {r['rank']} | {r['lora_M']} | {r['first_loss']} | {r['last_loss']} | {r['sec_per_step']} | {r['status']} |")
