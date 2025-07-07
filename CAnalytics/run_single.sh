#gcc -static multi_thread.cpp -lgsl -lgslcblas -lm -o a.out
g++ -fopenmp 1LDOS.cpp -lgsl -lgslcblas -lm -o a.out
./a.out


