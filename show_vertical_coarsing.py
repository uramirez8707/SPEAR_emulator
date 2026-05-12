import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

# Read in the pressures from the original file
file = "/scratch4/GFDL/gfdlscr/Uriel.Ramirez/archive_260429/RAW/18510101.full_state.ens_01.tile1.nc"
ds = xr.open_dataset(file)

pfull = ds['pfull'].values
phalf = ds['phalf'].values
dp = np.diff(phalf)
ds.close()

coarse_mapping = np.array([0]*7 + [1]*4 + [2]*3 + [3]*2 + [4]*3 + [5]*3 + [6]*3 + [7]*8)
ace_pressures = [25, 96, 203, 345, 517, 695, 847, 963]

fig, ax = plt.subplots(figsize=(15, 10))
colors = plt.cm.tab10(np.linspace(0, 1, 8)) # Retain distinct color scheme

# --- Left Side: Native Grid (33 Layers) ---
for i in range(33):
    bin_id = coarse_mapping[i]
    # Height is the actual mass (dp)
    ax.barh((phalf[i] + phalf[i+1])/2, 0.8, height=(phalf[i+1] - phalf[i]),
            color=colors[bin_id], alpha=0.3, edgecolor='black', linewidth=0.3)

# Add minimal midpoint dots for context
ax.scatter([0.4]*33, pfull, color='black', s=8, marker='_')

# --- Right Side: Coarsened Grid (8 Bins) ---
for bin_id in range(8):
    # Fixed mask calculation (from numpy array)
    indices = np.where(coarse_mapping == bin_id)[0]
    p_top, p_bot = phalf[indices[0]], phalf[indices[-1] + 1]

    # Simple rectangle visual, NO redundant labels here
    rect = plt.Rectangle((1.1, p_top), 0.2, p_bot - p_top, color=colors[bin_id], alpha=0.85)
    ax.add_patch(rect)

# 4. Global Explanation Elements (Moved lol)

# --- A. Formula Explanation (Top Right) ---
math_text = (
    r"Mass-Weighted Averaging (Ensures Conservation):" + "\n\n"
    r"$\bar{X}_{bin} = \frac{\sum (X_i \cdot \Delta p_i)}{\sum \Delta p_i}$" + "\n\n"
    r"$\Delta p_i = p_{half, i+1} - p_{half, i}$ (Pressure Weight)"
)
# Place formula in the top right
plt.figtext(0.70, 0.85, math_text, fontsize=12, horizontalalignment='left',
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=1'))

# --- B. Vertical Mapping Key (Legend) ---
# Consolidates all labeling into a single organized box
legend_elements = [
    plt.Line2D([0], [0], color=colors[i], lw=8, label=f'Bin {i} -> ML Label: {ace_pressures[i]} hPa')
    for i in range(8)
]
ax.legend(handles=legend_elements, loc='lower right',
          title="Vertical Coarsening Key", fontsize=11, title_fontsize=12, frameon=True)

# 5. Formatting
ax.set_yscale('log')
ax.set_ylim(phalf.max() + 10, phalf.min() - 0.1) # Invert axis: Surface to Top
ax.set_xlim(0, 1.8)
ax.set_ylabel("Pressure (hPa) - Log Scale", fontsize=12, fontweight='bold')
ax.set_title("SPEAR Vertical Coarsening Strategy", fontsize=16, fontweight='bold', pad=20)

# Custom X-Axis labels
ax.set_xticks([0.4, 1.2])
ax.set_xticklabels(["Original Grid\n(33 Native Layers)", "Coarsened Grid\n(8 ML Bins)"], fontsize=12, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)

plt.tight_layout()
plt.show()
