# Theme A ("typewriter lab"), matching SpinGlassBench docs/figure_style.md.
# Fixed meaning of color:
# RUST = focal/GPU series, COBALT = 16T Rayon / reference, JADE = 1T / SIMD, OCHRE = highlight;
# other series in graded grays with distinct markers.
PAPER = "#ffffff"; INK = "#2a2723"; INK2 = "#6b655c"; GRIDC = "#e2dacb"
RUST = "#c94f27"; COBALT = "#2f62b8"; JADE = "#069a8e"; OCHRE = "#d4960f"
G1 = "#4a463f"; G2 = "#8c857a"; G3 = "#b3ab9e"
S1 = "#86a0cf"; S2 = "#5a7ec0"; S3 = "#234a8c"; S4 = "#183361"
set border 1 lc rgb INK2 lw 1
set tics nomirror tc rgb INK2 scale 0.6
set ytics scale 0
set grid ytics lc rgb GRIDC lw 1 dt 3
set key noautotitle tc rgb INK spacing 1.2 samplen 2.5
set title tc rgb INK
set xlabel tc rgb INK
set ylabel tc rgb INK
c1 = RUST; c2 = COBALT; c3 = JADE; c4 = OCHRE; c5 = G2; c6 = G3; gray = INK2
set style line 1 lc rgb RUST lw 2 pt 7 ps 1.0
set style line 2 lc rgb COBALT lw 2 pt 5 ps 0.9
set style line 3 lc rgb JADE lw 2 pt 9 ps 0.9
set style line 4 lc rgb OCHRE lw 2 pt 13 ps 0.9
set style line 5 lc rgb G1 lw 1.6 pt 6 ps 0.9 dt 2
set style line 6 lc rgb G2 lw 1.5 pt 4 ps 0.8 dt 3
