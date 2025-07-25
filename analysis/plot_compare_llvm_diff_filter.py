import matplotlib.pyplot as plt
import numpy as np

# Data
instances = ['Bb1Inst2', 'Bb1Inst3', 'Bb1Inst4', 'Bb1Inst5', 'Bb1Inst6', 'Bb1Inst7', 'Bb1Inst8']
total = [590, 300, 429, 318, 350, 202, 137]
alltypes = [103, 154, 267, 222, 216, 181, 129]

# X-axis positions for the bars
x = np.arange(len(instances))

# Bar width
width = 0.35

# Create the figure and axis
fig, ax = plt.subplots()

# Plotting the bars
bars1 = ax.bar(x - width/2, total, width, label='Before Filter')
bars2 = ax.bar(x + width/2, alltypes, width, label='After Filter')

# Add labels, title, and ticks
ax.set_xlabel('Instance')
ax.set_ylabel('Values')
ax.set_title('Comparison of llvm-diff')
ax.set_xticks(x)
ax.set_xticklabels(instances)
ax.legend()

# Show the plot
plt.xticks(rotation=45)  # Rotate the x-axis labels if needed
plt.tight_layout()
plt.savefig("comparison.png")
