import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 15,
    'grid.color': '#e0e0e0',
    'grid.linestyle': '--',
    'grid.alpha': 0.7,
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle('OpenFold vs strux-rs Empirical Performance Comparison', fontweight='bold', y=0.98)

# 1. Superimposition Throughput
engines_1 = ['OpenFold\n(Biopython)', 'strux-rs\n(1T CPU)', 'strux-rs\n(16T CPU)', 'strux-rs\n(CUDA GPU)']
rates_1 = [203.4, 27067.5, 200423.5, 1411867.9]
speedups_1 = ['1.0x (Ref)', '133x', '985x', '6,941x']
colors_1 = ['#7f8c8d', '#2980b9', '#16a085', '#8e44ad']

bars1 = ax1.bar(engines_1, rates_1, color=colors_1, width=0.55, edgecolor='black', linewidth=0.8)
ax1.set_yscale('log')
ax1.set_ylim(50, 4e6)
ax1.set_ylabel('Throughput (Alignments / sec, log scale)', fontweight='bold')
ax1.set_title('Superimposition Throughput (1,919 atoms)', fontweight='bold', pad=12)
ax1.grid(True, which='both', axis='y')

for bar, rate, sp in zip(bars1, rates_1, speedups_1):
    h = bar.get_height()
    label = f'{rate:,.0f}/s\n({sp})'
    ax1.annotate(label, xy=(bar.get_x() + bar.get_width()/2, h),
                 xytext=(0, 5), textcoords="offset points",
                 ha='center', va='bottom', fontsize=9, fontweight='semibold')

# 2. MSA Parsing Throughput
engines_2 = ['OpenFold\n(Pure Python)', 'strux-rs\n(Zero-Copy FFI)']
rates_2 = [60731.9, 1199503.2]
speeds_mb = ['9.8 MB/s', '193.4 MB/s']
speedups_2 = ['1.0x (Ref)', '19.8x']
colors_2 = ['#7f8c8d', '#27ae60']

bars2 = ax2.bar(engines_2, rates_2, color=colors_2, width=0.45, edgecolor='black', linewidth=0.8)
ax2.set_ylim(0, 1.45e6)
ax2.set_ylabel('Throughput (Sequences / sec)', fontweight='bold')
ax2.set_title('A3M Alignment Parsing (25,000 seqs, 4.03 MB)', fontweight='bold', pad=12)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f'{x*1e-3:.0f}k' if x > 0 else '0'))
ax2.grid(True, axis='y')

for bar, rate, mb, sp in zip(bars2, rates_2, speeds_mb, speedups_2):
    h = bar.get_height()
    label = f'{rate:,.0f} seqs/s\n{mb} ({sp})'
    ax2.annotate(label, xy=(bar.get_x() + bar.get_width()/2, h),
                 xytext=(0, 6), textcoords="offset points",
                 ha='center', va='bottom', fontsize=9.5, fontweight='semibold')

plt.tight_layout()
os.makedirs('benchmarks', exist_ok=True)
out_local = 'benchmarks/openfold_vs_strux_comparison.png'
plt.savefig(out_local, dpi=300, bbox_inches='tight')

artifact_dir = r'C:\Users\Gebruiker\.gemini\antigravity\brain\533f8aaa-3288-419b-a6a1-a41762761b7f'
out_artifact = os.path.join(artifact_dir, 'openfold_vs_strux_comparison.png')
shutil.copy(out_local, out_artifact)
print('Saved plot to:', out_local, 'and', out_artifact)
