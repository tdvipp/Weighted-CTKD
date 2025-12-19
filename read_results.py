results_path = "/workspace/DSKD/outputs/tinyllama/tinyllama-1.1b-3T/wctkd/criterion=wctkd__forward_kl-lora-rank=256-alpha=8-dropout=0.1-bf16__teacher=mistral__kd^rate=0.5__kd^temp=2.0__wctkd^alpha=0.5__wctkd^beta=0.2__wctkd^gamma=0.3__wctkd^hidden_gamma=0.5__wctkd^top_k=8__epoch=15__bsz=4x2x1=8__lr=0.001/rougeL_results.jsonl"

import json

results = {}

with open(results_path, "r") as f:
    for line in f.readlines():
        try:
            result = json.loads(line)
            dataname = result["dataname"]
            rougeL = result["rougeL"]
            results[dataname] = results.get(dataname, []) + [rougeL]
        except:
            continue

for dataname in results:
    dataname_results = results[dataname]
    mean = sum(dataname_results) / len(dataname_results)
    print(dataname, f"{mean:.2f}")