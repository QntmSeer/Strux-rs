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

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.35, wspace=0.22)
fig.suptitle('Gold-Standard Biological Dataset Benchmark: strux-rs vs. Established Tools', 
             fontweight='bold', fontsize=14, y=0.98)

# ------------------------------------------------------------------------------
# PANEL A (Full Top Row): Ubiquitin 2K39 Ensemble (13,456 pairs)
# ------------------------------------------------------------------------------
ax1 = fig.add_subplot(gs[0, :])

packages_1 = [
    'Biopython\n(SVD)', 
    'SciPy\n(align)', 
    'MDAnalysis\n(QCPROT)', 
    'MDTraj\n(AVX QCP)', 
    'Gemmi\n(C++)', 
    'strux-rs\n(1T CPU)', 
    'Google JAX\n(vmap JIT)', 
    'ProDy\n(C-Kabsch)', 
    'strux-rs\n(16T CPU)',
    'strux-rs\n(CUDA GPU)'
]
rates_1 = [446.6, 5436.6, 10567.4, 26046.1, 39710.8, 43241.4, 76612.5, 92281.5, 596176.5, 1941698.0]
speedups_1 = ['1.0x', '12.2x', '23.7x', '58.3x', '88.9x', '96.8x', '172x', '207x', '1,335x', '4,348x']
colors_1 = [
    '#94a3b8', '#64748b', '#475569', '#334155', 
    '#334155', '#2563eb', '#1e293b', '#0f172a', '#1d4ed8', '#0f766e'
]

bars1 = ax1.bar(packages_1, rates_1, color=colors_1, width=0.62, edgecolor='#1e293b', linewidth=0.8)
ax1.set_yscale('log')
ax1.set_ylim(200, 6.0e6)
ax1.set_ylabel('Alignments / second (log scale)', fontweight='bold', fontsize=11)
ax1.set_title('A. Ubiquitin 2K39 Experimental NMR Ensemble (116 models, 13,456 pairs)', fontweight='bold', pad=10, fontsize=12)
ax1.grid(True, which='both', axis='y')
ax1.tick_params(axis='x', rotation=18, labelsize=9.5)

for bar, rate, sp in zip(bars1, rates_1, speedups_1):
    h = bar.get_height()
    label = f'{rate:,.0f}/s\n({sp})'
    ax1.annotate(label, xy=(bar.get_x() + bar.get_width()/2, h),
                 xytext=(0, 4), textcoords="offset points",
                 ha='center', va='bottom', fontsize=8.2, color='#111827')

# ------------------------------------------------------------------------------
# PANEL B (Bottom Left): Real GFP Trajectory Dynamics (trajectory.pdb, 1,919 atoms)
# ------------------------------------------------------------------------------
ax2 = fig.add_subplot(gs[1, 0])

packages_2 = ['MDAnalysis', 'MDTraj', 'strux-rs (SIMD)']
rates_2 = [336.9, 8912.2, 68770.8]
speedups_2 = ['1.0x (Ref)', '26.5x', '204x']
colors_2 = ['#475569', '#0f172a', '#2563eb']

bars2 = ax2.bar(packages_2, rates_2, color=colors_2, width=0.55, edgecolor='#1e293b', linewidth=0.8)
ax2.set_yscale('log')
ax2.set_ylim(100, 1.8e5)
ax2.set_ylabel('Frames / second (log scale)', fontweight='bold', fontsize=11)
ax2.set_title('B. GFP Dynamics Rg (1,919 atoms)', fontweight='bold', pad=10, fontsize=12)
ax2.grid(True, which='both', axis='y')
ax2.tick_params(axis='x', rotation=10, labelsize=9.5)

for bar, rate, sp in zip(bars2, rates_2, speedups_2):
    h = bar.get_height()
    label = f'{rate:,.0f} fps\n({sp})'
    ax2.annotate(label, xy=(bar.get_x() + bar.get_width()/2, h),
                 xytext=(0, 4), textcoords="offset points",
                 ha='center', va='bottom', fontsize=8.5, color='#111827')

# ------------------------------------------------------------------------------
# PANEL C (Bottom Right): Real Metagenomic MSA Ingestion (Throughput in MB/s)
# ------------------------------------------------------------------------------
ax3 = fig.add_subplot(gs[1, 1])

labels_3 = [
    'Kinase STO\nPure Python',
    'Kinase STO\nstrux-rs',
    'BFD A3M\nPure Python',
    'BFD A3M\nstrux-rs'
]
throughput_3 = [199.5, 270.7, 41.9, 168.9]  # MB/s
speedups_3 = ['1.0x (Ref)', '1.4x (72 ms)', '1.0x (Ref)', '4.0x (0.6 ms)']
colors_3 = ['#94a3b8', '#2563eb', '#94a3b8', '#2563eb']

bars3 = ax3.bar(labels_3, throughput_3, color=colors_3, width=0.55, edgecolor='#1e293b', linewidth=0.8)
ax3.set_ylim(0, 320)
ax3.set_ylabel('Parsing Throughput (MB / s)', fontweight='bold', fontsize=11)
ax3.set_title('C. Real Metagenomic MSA Ingestion', fontweight='bold', pad=10, fontsize=12)
ax3.grid(True, axis='y')
ax3.tick_params(axis='x', rotation=18, labelsize=9.5)

for bar, rate, sp in zip(bars3, throughput_3, speedups_3):
    h = bar.get_height()
    label = f'{rate:.1f} MB/s\n({sp})'
    ax3.annotate(label, xy=(bar.get_x() + bar.get_width()/2, h),
                 xytext=(0, 4), textcoords="offset points",
                 ha='center', va='bottom', fontsize=8.5, color='#111827')

out_local = 'benchmarks/gold_standard_comparison.png'
plt.savefig(out_local, dpi=300, bbox_inches='tight')

artifact_dir = r'C:\Users\Gebruiker\.gemini\antigravity\brain\533f8aaa-3288-419b-a6a1-a41762761b7f'
out_artifact = os.path.join(artifact_dir, 'gold_standard_comparison.png')
shutil.copy(out_local, out_artifact)

print("Saved gold standard plot to:", out_local, "and", out_artifact)