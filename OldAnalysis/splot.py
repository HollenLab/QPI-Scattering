import matplotlib.pyplot as plt

bias = [0.1, 0.08, 0.12]
k = [2.402, 2.350, 2.373]

plt.scatter(k, bias)
plt.ylabel("Bias (V)")
plt.xlabel("Wavenumber (1/nm)")