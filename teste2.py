import numpy as np

a = np.array([[1,2,3],[4,5,6],[7,8,9]]).flatten()
# a = a.reshape(1,3,3)
print(f"a = {a}")



# print(a)

# b = np.hstack((np.array([0,0]),a.flatten())).T
# c = np.array([])
# c = b
# c = np.vstack((c,b))

# print("aaaaa")
# print(c)
# print("aaaaa")
# print(f"c.T = {c.T}")


d = []
d.append(np.random.randint(1,10)*a)
d.append(np.random.randint(1,10)*a)
d.append(np.random.randint(1,10)*a)
d = np.array(d).T
# d = np.reshape(d,(-1,3,3))
print(f"d[2] = {d[2]}")
print("d = {}".format(d))
print("d.shape = {}".format(d.shape))

# #JC
# g = np.array([])
# # g = []
# for i in range(4):
#     f = np.random.randint(1,10)*a
#     g = np.append(g,f)
# print(g)
# # g = g.reshape(-1, 3, 3)
# print(g[2])
