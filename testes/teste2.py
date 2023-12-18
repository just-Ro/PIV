import numpy as np

features = 10

H = [[0 for i in range(features)] for j in range (features)]


for i in range(features-1):
    # upper triangular diagonal element
    H[i][i+1] = 1
    
    # lower triangular diagonal element
    H[i+1][i] = -H[i][i+1]
    
    # i = 1, j = 0
    # H02 = H12.H01
    for j in range(i-1,-1,-1):
        # upper triangular elements above
        H[j][i+1] = H[j+1][i+1] + H[j][i]

        # lower triangular elements to the left
        H[i+1][j] = H[i+1][j+1] + H[i][j]

print(np.array(H))