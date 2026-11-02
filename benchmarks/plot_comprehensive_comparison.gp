# Gnuplot script: Multi-Dimensional Performance Scaling Benchmark (SpinGlassBench Theme A)
# Usage: gnuplot benchmarks/plot_comprehensive_comparison.gp

load "benchmarks/style.gp"
set terminal pngcairo size 1650,480 enhanced font "Latin Modern Mono,9" background rgb PAPER
set output 'benchmarks/openfold_vs_strux_scaling_gnuplot.png'

set multiplot layout 1,3

# Panel 1: Depth Scaling
set title "A3M parse time stays sub-50ms up to 10k depth\n{/*0.8 L = 150 residues; zero-copy FFI vs pure Python}" offset 0,0.3
set xlabel 'alignment depth (sequences)'
set ylabel 'parse time (seconds)'
set key top left

plot 'benchmarks/a3m_depth_scaling.dat' using 1:3 with linespoints ls 5 title 'OpenFold (Python)', \
     'benchmarks/a3m_depth_scaling.dat' using 1:4 with linespoints ls 1 title 'strux-rs (Zero-Copy FFI)'

# Panel 2: Length Scaling (MB/s)
set title "Zero-copy FFI sustains >150 MB/s across lengths\n{/*0.8 N = 10,000 sequences; L = 50 to 1,000 residues}" offset 0,0.3
set xlabel 'sequence length (residues)'
set ylabel 'ingestion rate (MB / s)'
set yrange [0:350]
set key top right

plot 'benchmarks/a3m_length_scaling.dat' using 1:5 with linespoints ls 5 title 'OpenFold (Python)', \
     'benchmarks/a3m_length_scaling.dat' using 1:6 with linespoints ls 2 title 'strux-rs (Zero-Copy FFI)'

# Panel 3: Superimposition vs Protein Size (log scale)
set title "1T Rust Kabsch is 97x faster than Bio.SVD\n{/*0.8 N = 100 to 10,000 atoms; single-thread CPU parity}" offset 0,0.3
set xlabel 'target protein size (atoms)'
set ylabel 'alignments / second (log scale)'
set logscale y 10
set format y '1e%T'
set yrange [50:500000]
set key top right

plot 'benchmarks/superimpose_atom_scaling.dat' using 1:4 with linespoints ls 5 title 'OpenFold (Bio.SVD)', \
     'benchmarks/superimpose_atom_scaling.dat' using 1:5 with linespoints ls 3 title 'strux-rs (1T Kabsch)'

unset multiplot
set output
