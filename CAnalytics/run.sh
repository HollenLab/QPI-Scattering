#gcc -static multi_thread.cpp -lgsl -lgslcblas -lm -o a.out
g++ -fopenmp 2LDOS.cpp -lgsl -lgslcblas -lm -o a.out
./a.out
python sum_data.py
python sum_plot.py


