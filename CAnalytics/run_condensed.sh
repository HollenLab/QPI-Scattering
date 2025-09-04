#gcc -static multi_thread.cpp -lgsl -lgslcblas -lm -o a.out
g++ -fopenmp 2LDOS.cpp -lgsl -lgslcblas -lm -o a.out
./a.out
python condensed_plot.py

