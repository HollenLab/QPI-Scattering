#gcc -static multi_thread.cpp -lgsl -lgslcblas -lm -o a.out
g++ -fopenmp poster_plot.cpp -lgsl -lgslcblas -lm -o a.out
./a.out


