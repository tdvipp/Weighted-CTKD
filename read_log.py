import re
import csv

filename = "tiny-train.log"
output_csv = "tiny-train.csv"

# Patterns to match training and validation log lines
train_pattern = r'train \| epoch (\d+):\s+(\d+) / \s+(\d+)\s+global_step=(\d+), loss=([\d.]+), nll_loss=([\d.]+), wctkd_loss=([\d.]+), dskd_loss=([\d.]+), accuracy=([\d.]+), micro_step_time=([\d.]+), step_time=([\d.]+), t2s_ce_loss=([\d.]+), t2s_acc=([\d.]+), max_t2s_prob=([\d.]+), t2s_kd_loss=([\d.]+), s2t_kd_loss=([\d.]+), s2t_acc=([\d.]+), lr=([\d.e-]+), projector_lr=([\d.e-]+), scale=([\d.]+)'
dev_pattern = r'dev \| .*? \| \{.*?\'rougeL\': ([\d.]+)\}'

# Store training data and validation data separately
training_data = []
validation_data = {}  # epoch -> rougeL

current_epoch = None

with open(filename, "r") as f:
    for line in f:
        # Match training lines
        train_match = re.search(train_pattern, line)
        if train_match:
            epoch = int(train_match.group(1))
            step = int(train_match.group(2))
            total_steps = int(train_match.group(3))
            global_step = int(train_match.group(4))
            loss = float(train_match.group(5))
            current_epoch = epoch
            
            training_data.append({
                'epoch': epoch,
                'global_step': global_step,
                'loss': loss,
                'rougeL': None  # Will be filled later
            })
        
        # Match validation lines
        dev_match = re.search(dev_pattern, line)
        if dev_match:
            rougeL = float(dev_match.group(1))
            if current_epoch is not None:
                validation_data[current_epoch] = rougeL

# Merge validation data into training data
for entry in training_data:
    epoch = entry['epoch']
    if epoch in validation_data:
        entry['rougeL'] = validation_data[epoch]

# Write to CSV
if training_data:
    fieldnames = ['epoch', 'global_step', 'loss', 'rougeL']
    
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(training_data)
    
    print(f"Extracted {len(training_data)} training records and saved to {output_csv}")
    print(f"Found {len(validation_data)} validation records")
else:
    print("No training log entries found!")