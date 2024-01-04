import numpy as np
from pprint import pprint

r = "\033[91m██\033[0m"
g = "\033[92m██\033[0m"
y = "\033[93m██\033[0m"
b = "  "

size = 100
matrix = [[b for i in range(size)] for j in range(size)]
for i in range(size):
    matrix[i][i] = g

for i in range(size):
    for j in range(size):
        print(matrix[i][j], end="")
    print()
