import numpy as np

a = np.arange(0, 10)
b = np.arange(0, 20)
arr = np.setdiff1d(b, a)
print(arr)