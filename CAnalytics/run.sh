#gcc -static multi_thread.cpp -lgsl -lgslcblas -lm -o a.out
g++ -fopenmp LDOS_Final.cpp -lgsl -lgslcblas -lm -o a.out
./a.out
python condensed_plot_phase.py


