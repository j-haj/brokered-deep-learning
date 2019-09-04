set terminal pdfcairo enhanced color
set output "fuzz5.pdf"
set style data lines
set xlabel "Generation"
set ylabel "Fitness"
set title "Average Population Fitness Over Time"
set key off
set datafile separator ","

plot "./fuzz_5epochs.out" using 1:3:4 with filledcurves fill solid .25 fc rgb "red", \
     "./fuzz_5epochs.out" using 1:2 with lines lc rgb "red"