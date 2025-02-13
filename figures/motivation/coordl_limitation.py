import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
# Data
runs = ["ResNet-18", "ResNet-50", "ResNet-101", "ResNet-152"]
# potential_time = np.array([2.14, 3.06, 5.04, 7.64])  # Rounded values
# actual_time = np.array([7.64, 7.64, 7.64, 7.64])

potential_time = np.array([714.285, 500, 303, 200])  # Rounded values
actual_time = np.array([199.998981823494, 199.998545465253, 199.998545465253, 199.998545465253])

total_delay = actual_time - potential_time
line_color = 'black'

visual_map = {
    'Potential': {'color': '#005250', 'hatch': '//', 'edgecolor': 'black', 'alpha': 1.0, 'marker':'o', 'linestyle':'-'},
    'Actual': {'color': '#FEA400', 'hatch': '\\\\', 'edgecolor': 'black', 'alpha': 1.0,  'marker':'o', 'linestyle':'-'},
}


# Bar width and positions
bar_width = 0.35
x = np.arange(len(runs))

# Create figure
fig, ax = plt.subplots(figsize=(6, 4))

# Plot bars side by side
bars_potential = ax.bar(x - bar_width/2, potential_time, bar_width, label="Potential", 
                        color=visual_map['Potential']['color'], 
                        edgecolor=visual_map['Potential']['edgecolor'],
                        hatch=visual_map['Potential']['hatch'],
                        alpha=visual_map['Potential']['alpha'])
                    

bars_actual = ax.bar(x + bar_width/2, actual_time, bar_width, 
                     label="Actual", 
                        color=visual_map['Actual']['color'],
                        edgecolor=visual_map['Actual']['edgecolor'],
                        hatch=visual_map['Actual']['hatch'],
                        alpha=visual_map['Actual']['alpha'])


# # Draw red vertical lines with horizontal caps
# for i in range(len(runs)-1):
#     # Vertical delay line
#     ax.plot([x[i] - bar_width/2, x[i] - bar_width/2],  
#             [potential_time[i], actual_time[i]],  
#             color=line_color, linestyle='-', linewidth=1)

#     # Horizontal caps
#     cap_width = 0.15  # Length of the horizontal cap
#     ax.plot([x[i] - bar_width/2 - cap_width, x[i] - bar_width/2 + cap_width], 
#             [potential_time[i], potential_time[i]], color=line_color, linewidth=1)  # Bottom cap
#     ax.plot([x[i] - bar_width/2 - cap_width, x[i] - bar_width/2 + cap_width], 
#             [actual_time[i], actual_time[i]], color=line_color, linewidth=1)  # Top cap

# Labels and legend
# ax.set_ylabel("Training Time (hours)")
ax.set_ylabel("Throuhgput (Batches/Second)")

ax.set_xticks(x)
ax.set_xticklabels(runs)
ax.set_xlabel("Model")
# ax.set_ylim([0, 9])
ax.set_ylim([0, 750])

# Custom legend entries for bars and delay line
# delay_line = Line2D([0], [0], color=line_color, linewidth=1, label="Delay")

# Adding bars to the legend
# ax.legend(handles=[bars_potential, bars_actual, delay_line], loc="upper center", ncol=3, frameon=True)
ax.legend(handles=[bars_potential, bars_actual], loc="upper center", ncol=2, frameon=True)


# Grid for readability
ax.yaxis.grid(True, linestyle="--", alpha=0.6)

# Show plot
plt.tight_layout()
plt.show()
