import numpy as np

# Assuming len(features) = 4
features = [1, 2, 3, 4]

aux = np.zeros(len(features) - 1)
for i in range(len(features) - 1):
    aux[i] = i + 1

print(aux.shape)
print(aux)