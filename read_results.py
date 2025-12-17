results_path = "/workspace/DSKD/outputs/gpt2/gpt2-base/wctkd/criterion=wctkd__forward_kl-bf16__teacher=Qwen1.5-1.8B__kd^rate=0.5__kd^temp=2.0__wctkd^alpha=0.5__wctkd^beta=0.2__wctkd^gamma=0.3__wctkd^hidden_gamma=0.5__wctkd^top_k=8__epoch=20__bsz=4x2x1=8__lr=0.0005__proj^lr=0.001/rougeL_results.jsonl"

import json

results = {}

with open(results_path, "r") as f:
    for line in f.readlines():
        result = json.loads(line)
        dataname = result["dataname"]
        rougeL = result["rougeL"]
        results[dataname] = results.get(dataname, []) + [rougeL]

for dataname in results:
    dataname_results = results[dataname]
    mean = sum(dataname_results) / len(dataname_results)
    print(dataname, f"{mean:.2f}")