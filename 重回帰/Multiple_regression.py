import numpy as np
# print(df)
# print(data.target)

Y = np.array(y)
X = np.array([x1,x2]) #説明変数はいくらでも

e = np.vstack([np.ones(X.shape[1]), X]) #縦に重ねる
b, a1,a2 = np.linalg.lstsq(e.T, Y)[0]

print(b,a1,a2)
