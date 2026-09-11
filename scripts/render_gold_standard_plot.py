import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

plt.style.use('default')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 10,
    'axes.edgecolor': '#222222',
    'axes.linewidth': 0.8,
    'grid.color': '#e5e7eb',
    'grid.linestyle': '--',
    'grid.alpha': 0.8,
    'axes.labelcolor': '#111827',
    'xtick.color': '#111827',
    'ytick.color': '#111827',
    'figure.titlesize': 13,
    'figure.facecolor': '#ffffff',
    'axes.facecolor': '#ffffff',
})

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5), gridspec_kw={'width_ratios': [2.4, 0.9, 0.8]})
fig.suptitle('Gold-Standard Biological Dataset Benchmark: strux-rs vs. Established Tools', 
             fontweight='bold', fontsize=13, y=0.98)

# ------------------------------------------------------------------------------
# PANEL 1: Ubiquitin 2K39 Ensemble All-to-All Superimposition (13,456 pairs)
# ------------------------------------------------------------------------------
packages_1 = [
    'Biopython\n(SVD)', 
    'OpenFold/\nAlphaFold', 
    'SciPy\n(align)', 
    'MDAnalysis\n(QCPROT)', 
    'MDTraj\n(AVX QCP)', 
    'Gemmi\n(C++)', 
    'strux-rs\n(1T CPU)', 
    'Google JAX\n(vmap JIT)', 
    'ProDy\n(C-Kabsch)', 
    'strux-rs\n(16T CPU)'
]
rates_1 = [446.6, 451.1, 5436.6, 10567.4, 26046.1, 39710.8, 43241.4, 76612.5, 92281.5, 596176.5]
speedups_1 = ['1.0x', '1.0x', '12.2x', '23.7x', '58.3x', '88.9x', '96.8x', '172x', '207x', '1,335x']
colors_1 = [
    '#94a3b8', '#94a3b8', '#64748b', '#475569', '#334155', 
    '#334155', '#2563eb', '#1e293b', '#0f172a', '#1d4ed8'
]

bars1 = ax1.bar(packages_1, rates_1, color=colors_1, width=0.62, edgecolor='#1e293b', linewidth=0.8)
ax1.set_yscale('log')
ax1.set_ylim(200, 1.5e6)
ax1.set_ylabel('Alignments / second (log scale)', fontweight='bold')
ax1.set_title('A. Ubiquitin 2K39 Ensemble (116 models, 13.5k pairs)', fontweight='bold', pad=10, fontsize=11)
ax1.grid(True, which='both', axis='y')
ax1.tick_params(axis='x', rotation=25)

for bar, rate, sp in zip(bars1, rates_1, speedups_1):
    h = bar.get_height()
    label = f'{rate:,.0f}/s\n({sp})'
    ax1.annotate(label, xy=(bar.get_x() + bar.get_width()/2, h),
                 xytext=(0, 4), textcoords="offset points",
                 ha='center', va='bottom', fontsize=7.5, color='#111827')

# ------------------------------------------------------------------------------
# PANEL 2: Real GFP Trajectory Rg Dynamics (trajectory.pdb, 1,919 atoms)
# ------------------------------------------------------------------------------
packages_2 = ['MDAnalysis', 'MDTraj', 'strux-rs (SIMD)']
rates_2 = [336.9, 8912.2, 68770.8]
speedups_2 = ['1.0x (Ref)', '26.5x', '204x']
colors_2 = ['#475569', '#0f172a', '#2563eb']

bars2 = ax2.bar(packages_2, rates_2, color=colors_2, width=0.55, edgecolor='#1e293b', linewidth=0.8)
ax2.set_yscale('log')
ax2.set_ylim(100, 1.5e5)
ax2.set_ylabel('Frames / second (log scale)', fontweight='bold')
ax2.set_title('B. GFP Dynamics Rg (1,919 atoms)', fontweight='bold', pad=10, fontsize=11)
ax2.grid(True, which='both', axis='y')
ax2.tick_params(axis='x', rotation=12)

for bar, rate, sp in zip(bars2, rates_2, speedups_2):
    h = bar.get_height()
    label = f'{rate:,.0f} fps\n({sp})'
    ax2.annotate(label, xy=(bar.get_x() + bar.get_width()/2, h),
                 xytext=(0, 4), textcoords="offset points",
                 ha='center', va='bottom', fontsize=8, color='#111827')

# ------------------------------------------------------------------------------
# PANEL 3: Real Production Stockholm Ingestion (hmm_output.sto, 19.5 MB)
# ------------------------------------------------------------------------------
packages_3 = ['Google AF / OF\n(Pure Python)', 'strux-rs\n(Zero-Copy FFI)']
rates_3 = [199.5, 270.7]
speedups_3 = ['1.0x (Ref)', '1.4x (271 MB/s)']
colors_3 = ['#94a3b8', '#2563eb']

bars3 = ax3.bar(packages_3, rates_3, color=colors_3, width=0.48, edgecolor='#1e293b', linewidth=0.8)
ax3.set_ylim(0, 350)
ax3.set_ylabel('Throughput (MB / s)', fontweight='bold')
ax3.set_title('C. Real Kinase Stockholm (19.5 MB, 30k seqs)', fontweight='bold', pad=10, fontsize=11)
ax3.grid(True, axis='y')

for bar, rate, sp in zip(bars3, rates_3, speedups_3):
    h = bar.get_height()
    label = f'{rate:.1f} MB/s\n({sp})'
    ax3.annotate(label, xy=(bar.get_x() + bar.get_width()/2, h),
                 xytext=(0, 5), textcoords="offset points",
                 ha='center', va='bottom', fontsize=8.5, color='#111827')

plt.tight_layout()

out_local = 'benchmarks/gold_standard_comparison.png'
plt.savefig(out_local, dpi=300, bbox_inches='tight')

artifact_dir = r'C:\Users\Gebruiker\.gemini\antigravity\brain\533f8aaa-3288-419b-a6a1-a41762761b7f'
out_artifact = os.path.join(artifact_dir, 'gold_standard_comparison.png')
shutil.copy(out_local, out_artifact)

print("Saved gold standard plot to:", out_local, "and", out_artifact)