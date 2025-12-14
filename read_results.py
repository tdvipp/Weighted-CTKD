results_path = "/workspace/DSKD/outputs/gpt2/gpt2-base/multiot/rougeL_results.jsonl"

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