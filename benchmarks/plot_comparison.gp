# Gnuplot script: OpenFold vs strux-rs Performance Comparison
# Run with: gnuplot benchmarks/plot_comparison.gp

set terminal pngcairo size 1280,640 enhanced font 'Arial,12' fontscale 1.0
set output 'benchmarks/openfold_vs_strux_comparison.png'

set multiplot layout 1,2 title 'OpenFold vs strux-rs Empirical Performance Comparison' font 'Arial-Bold,15'

# ==============================================================================
# SUBPLOT 1: Structural Superimposition Throughput (Log Scale)
# ==============================================================================
set title 'Kabsch Superimposition Throughput (1,919 atoms)' font 'Arial-Bold,12'
set style data histogram
set style histogram cluster gap 1
set style fill solid 0.85 border -1
set boxwidth 0.6
set grid ytics lc rgb '#dcdcdc' lt 1 lw 1

set logscale y 10
set format y '10^{%T}'
set ylabel 'Throughput (Alignments / sec, log scale)' offset 1,0 font 'Arial-Bold,11'
set yrange [50:3000000]

set xtics rotate by -15 scale 0 font 'Arial,10'
set key off

plot 'benchmarks/superimposition.dat' using 2:xtic(1) lc rgb '#2b5c8f', \
     '' using 0:2:(sprintf('%.0f/s\n(%gx)', , )) with labels font 'Arial,9' offset 0,1.2 textcolor rgb '#111111'

# ==============================================================================
# SUBPLOT 2: A3M MSA Parsing Throughput (Linear Scale)
# ==============================================================================
unset logscale y
set format y '%g'
set title 'A3M Alignment Parsing (25,000 seqs, 4.03 MB)' font 'Arial-Bold,12'
set ylabel 'Throughput (Sequences / sec)' offset 1,0 font 'Arial-Bold,11'
set yrange [0:1400000]
set ytics 200000
set xtics rotate by 0 font 'Arial,10'

plot 'benchmarks/msa_parsing.dat' using 2:xtic(1) lc rgb '#2e7d32', \
     '' using 0:2:(sprintf('%.0f seqs/s\n(%.1f MB/s, %gx)', , , )) with labels font 'Arial,9' offset 0,1.2 textcolor rgb '#111111'

unset multiplot
set output
