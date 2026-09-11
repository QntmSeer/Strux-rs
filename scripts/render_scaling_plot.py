import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 15,
    'grid.color': '#e5e5e5',
    'grid.linestyle': '--',
    'grid.alpha': 0.8,
})

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5))
fig.suptitle('OpenFold vs strux-rs Multi-Dimensional Performance Scaling Benchmark', fontweight='bold', y=0.98)

# Panel 1: A3M Depth Scaling
depths = [1000, 2500, 5000, 10000, 25000, 50000]
of_time_depth = [0.0158, 0.0364, 0.0781, 0.1516, 0.3928, 0.6280]
st_time_depth = [0.0019, 0.0032, 0.0056, 0.0119, 0.0305, 0.0398]

ax1.plot(depths, of_time_depth, 'o-', color='#7f8c8d', lw=2.2, ms=6, label='OpenFold (Pure Python)')
ax1.plot(depths, st_time_depth, 's-', color='#27ae60', lw=2.2, ms=6, label='strux-rs (Zero-Copy FFI)')
ax1.set_xlabel('Alignment Depth (Sequences)', fontweight='bold')
ax1.set_ylabel('Parse Time (seconds)', fontweight='bold')
ax1.set_title('A. A3M Parsing Time vs Depth (L=150)', fontweight='bold', pad=10)
ax1.grid(True)
ax1.legend(frameon=True, facecolor='white', framealpha=0.9)

# Annotate speedup on max depth
ax1.annotate('15.8x faster\n(0.040s vs 0.628s)', xy=(50000, 0.0398),
             xytext=(32000, 0.25),
             arrowprops=dict(facecolor='#27ae60', shrink=0.08, width=1.5, headwidth=6),
             fontsize=9.5, fontweight='bold', color='#1e7e34')

# Panel 2: A3M Length Scaling (MB/s)
lengths = [50, 100, 250, 500, 1000]
of_mbs = [9.5, 13.3, 15.6, 17.9, 17.6]
st_mbs = [193.0, 235.4, 283.3, 295.7, 278.6]

ax2.plot(lengths, of_mbs, 'o-', color='#7f8c8d', lw=2.2, ms=6, label='OpenFold (Pure Python)')
ax2.plot(lengths, st_mbs, '^-', color='#2980b9', lw=2.2, ms=6, label='strux-rs (Zero-Copy FFI)')
ax2.set_xlabel('Sequence Length (Residues)', fontweight='bold')
ax2.set_ylabel('Data Throughput (MB / s)', fontweight='bold')
ax2.set_title('B. Ingestion Throughput vs Length (N=10k)', fontweight='bold', pad=10)
ax2.grid(True)
ax2.legend(frameon=True, facecolor='white', framealpha=0.9)

# Annotate speedup
ax2.annotate('~290 MB/s sustained\n(16x - 20x speedup)', xy=(500, 295.7),
             xytext=(300, 200),
             arrowprops=dict(facecolor='#2980b9', shrink=0.08, width=1.5, headwidth=6),
             fontsize=9.5, fontweight='bold', color='#1b4f72')

# Panel 3: Superimposition Throughput vs Atom Count
atoms = [100, 300, 1000, 1919, 3500]
of_rate = [4085.0, 1663.0, 530.0, 278.0, 153.0]
st_rate = [190464.0, 187575.0, 69401.0, 35183.0, 19814.0]

ax3.plot(atoms, of_rate, 'o-', color='#7f8c8d', lw=2.2, ms=6, label='OpenFold (Bio.SVDSuperimposer)')
ax3.plot(atoms, st_rate, 'd-', color='#8e44ad', lw=2.2, ms=6, label='strux-rs (Single-Thread SVD)')
ax3.set_yscale('log')
ax3.set_xlabel('Target Protein Size (Atoms)', fontweight='bold')
ax3.set_ylabel('Throughput (Alignments / sec, log scale)', fontweight='bold')
ax3.set_title('C. Kabsch Superimposition vs Protein Size', fontweight='bold', pad=10)
ax3.grid(True, which='both')
ax3.legend(frameon=True, facecolor='white', framealpha=0.9)

# Annotate speedup
ax3.annotate('130x speedup\n(35,183/s vs 278/s)', xy=(1919, 35183.0),
             xytext=(1200, 3500),
             arrowprops=dict(facecolor='#8e44ad', shrink=0.08, width=1.5, headwidth=6),
             fontsize=9.5, fontweight='bold', color='#512e5f')

plt.tight_layout()

# Save locally
out_local = 'benchmarks/openfold_vs_strux_scaling.png'
plt.savefig(out_local, dpi=300, bbox_inches='tight')

# Save to brain artifact
artifact_dir = r'C:\Users\Gebruiker\.gemini\antigravity\brain\533f8aaa-3288-419b-a6a1-a41762761b7f'
out_artifact = os.path.join(artifact_dir, 'openfold_vs_strux_scaling.png')
shutil.copy(out_local, out_artifact)

# Save to OpenFold fork docs/imgs
openfold_img_dir = r'c:\Users\Gebruiker\Documents\Computational Bio\openfold\docs\imgs'
os.makedirs(openfold_img_dir, exist_ok=True)
out_of = os.path.join(openfold_img_dir, 'openfold_vs_strux_scaling.png')
shutil.copy(out_local, out_of)

print(f'Successfully rendered and saved to:')
print('  1.', out_local)
print('  2.', out_artifact)
print('  3.', out_of)
