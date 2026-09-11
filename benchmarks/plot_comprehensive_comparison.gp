# Gnuplot script: Multi-Dimensional Performance Scaling Benchmark
# Usage: gnuplot benchmarks/plot_comprehensive_comparison.gp

set terminal pngcairo size 1800,550 enhanced font 'Arial,11'
set output 'benchmarks/openfold_vs_strux_scaling_gnuplot.png'

set multiplot layout 1,3 title 'OpenFold vs strux-rs Multi-Dimensional Performance Scaling Benchmark' font 'Arial-Bold,15'

# Panel 1: Depth Scaling
set title 'A. A3M Parsing Time vs Depth (L=150)' font 'Arial-Bold,12'
set xlabel 'Alignment Depth (Sequences)' font 'Arial-Bold,10'
set ylabel 'Parse Time (seconds)' font 'Arial-Bold,10'
set grid
set key top left

plot 'benchmarks/a3m_depth_scaling.dat' using 1:3 with linespoints lw 2 pt 7 lc rgb '#7f8c8d' title 'OpenFold (Pure Python)', \
     'benchmarks/a3m_depth_scaling.dat' using 1:4 with linespoints lw 2 pt 5 lc rgb '#27ae60' title 'strux-rs (Zero-Copy FFI)'

# Panel 2: Length Scaling (MB/s)
set title 'B. Ingestion Throughput vs Length (N=10k)' font 'Arial-Bold,12'
set xlabel 'Sequence Length (Residues)' font 'Arial-Bold,10'
set ylabel 'Data Throughput (MB / s)' font 'Arial-Bold,10'
set yrange [0:350]
set key top left

plot 'benchmarks/a3m_length_scaling.dat' using 1:5 with linespoints lw 2 pt 7 lc rgb '#7f8c8d' title 'OpenFold (Pure Python)', \
     'benchmarks/a3m_length_scaling.dat' using 1:6 with linespoints lw 2 pt 9 lc rgb '#2980b9' title 'strux-rs (Zero-Copy FFI)'

# Panel 3: Superimposition vs Protein Size (log scale)
set title 'C. Kabsch Superimposition vs Protein Size' font 'Arial-Bold,12'
set xlabel 'Target Protein Size (Atoms)' font 'Arial-Bold,10'
set ylabel 'Throughput (Alignments / sec, log scale)' font 'Arial-Bold,10'
set logscale y 10
set format y '10^{%T}'
set yrange [50:500000]
set key top right

plot 'benchmarks/superimpose_atom_scaling.dat' using 1:4 with linespoints lw 2 pt 7 lc rgb '#7f8c8d' title 'OpenFold (Bio.SVDSuperimposer)', \
     'benchmarks/superimpose_atom_scaling.dat' using 1:5 with linespoints lw 2 pt 11 lc rgb '#8e44ad' title 'strux-rs (Single-Thread SVD)'

unset multiplot
set output
