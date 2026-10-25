import argparse, csv, os, re
p = argparse.ArgumentParser()
p.add_argument("--tag", required=True)
p.add_argument("--log", required=True)
p.add_argument("--rank", required=True)
p.add_argument("--bs", required=True)
p.add_argument("--seq", required=True)
p.add_argument("--csv", default="experiments.csv")
a = p.parse_args()

txt = open(a.log, errors="ignore").read()
losses = [float(x) for x in re.findall(
    r"Epoch:\[\d+/\d+\]\(\d+/\d+\), loss:\s*([0-9.]+)", txt)]
lora = re.search(r"LoRA 参数量:\s*([0-9.]+)\s*M", txt)
elapsed = re.search(r"elapsed[= ](\d+)", txt)
status = "OK" if ("Traceback" not in txt and "HSA exception" not in txt) else "FAIL"
last = round(sum(losses[-10:]) / max(len(losses[-10:]), 1), 4) if losses else ""
first = losses[0] if losses else ""
per_step = round(int(elapsed.group(1)) / max(len(losses), 1), 2) if elapsed and losses else ""

row = dict(tag=a.tag, rank=a.rank, bs=a.bs, seq=a.seq,
           lora_M=lora.group(1) if lora else "",
           first_loss=first, last_loss=last, sec_per_step=per_step, status=status)

exists = os.path.exists(a.csv)
with open(a.csv, "a", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(row.keys()))
    if not exists: w.writeheader()
    w.writerow(row)
print("recorded:", row)
