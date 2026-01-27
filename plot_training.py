import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 7)

# Read the CSV file
csv_file = "gpt2-base-train.csv"
df = pd.read_csv(csv_file)

# Get epoch boundaries (last global_step for each epoch)
epoch_boundaries = df.groupby('epoch')['global_step'].max().reset_index()
epoch_boundaries.columns = ['epoch', 'global_step']

# Get rougeL for each epoch (should be same for all steps in epoch)
epoch_rougeL = df.groupby('epoch').agg({
    'rougeL': 'first'
}).reset_index()

# Merge to get rougeL at epoch boundaries
epoch_data = epoch_boundaries.merge(epoch_rougeL, on='epoch')
epoch_data = epoch_data[epoch_data['rougeL'].notna()]

# Create the plot with two y-axes
fig, ax1 = plt.subplots(figsize=(14, 7))

# Plot training loss on left y-axis (blue line) - by global_step
color1 = '#2E86AB'
ax1.set_xlabel('Global Step', fontsize=12, fontweight='bold')
ax1.set_ylabel('Loss', fontsize=12, fontweight='bold', color=color1)
ax1.plot(df['global_step'], df['loss'], linewidth=2, label='Training Loss', color=color1)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True, alpha=0.3, linestyle='--', axis='both')

# Add vertical lines to mark epoch boundaries
for _, row in epoch_boundaries.iterrows():
    ax1.axvline(x=row['global_step'], color='gray', linestyle='--', alpha=0.5, linewidth=1)

# Create second y-axis for rougeL
ax2 = ax1.twinx()
color2 = '#DC143C'  # Red color for RougeL
ax2.set_ylabel('RougeL Score', fontsize=12, fontweight='bold', color=color2)

# Plot rougeL at epoch boundaries (end of each epoch) as red line with dots
ax2.plot(epoch_data['global_step'], epoch_data['rougeL'], 
         marker='o', markersize=8, linewidth=2, label='RougeL Score', 
         color=color2, alpha=0.9)
ax2.tick_params(axis='y', labelcolor=color2)

# Add text annotations above each rougeL point
for _, row in epoch_data.iterrows():
    # Calculate offset to place text above the point
    y_offset = (ax2.get_ylim()[1] - ax2.get_ylim()[0]) * 0.02  # 2% of y-axis range
    ax2.text(row['global_step'], row['rougeL'] + y_offset, f"{row['rougeL']:.2f}", 
             ha='center', va='bottom', fontsize=9, color=color2, fontweight='bold')

# Add title
plt.title('Loss and RougeL Score', fontsize=14, fontweight='bold', pad=20)

# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=11, loc='upper left')

# Format x-axis to show steps nicely
ax1.ticklabel_format(style='plain', axis='x')

# Tight layout for better spacing
plt.tight_layout()

# Save the plot
output_file = "training_loss_plot.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Plot saved to {output_file}")
