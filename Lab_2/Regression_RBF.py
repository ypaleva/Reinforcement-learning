import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

rawData = np.genfromtxt('housing.txt')
N, pp1 = rawData.shape
# Last column is target
X = np.matrix(rawData[:, 0:pp1 - 1])
y = np.matrix(rawData[:, pp1 - 1]).T
print(X.shape, y.shape)
print(y)

# Solve linear regression, plot target and prediction
w = (np.linalg.inv(X.T * X)) * X.T * y
yh_lin = X * w
plt.plot(y, yh_lin, '.', color='blue')
plt.xlabel('Target')
plt.ylabel('Predicted')
plt.title('Linear Regression')
plt.show()

# J = 20 basis functions obtained by k-means clustering
# sigma set to standard deviation of entire data
J = 50
kmeans = KMeans(n_clusters=J, random_state=0).fit(X)
sig = np.std(X)
# Construct design matrix
U = np.zeros((N, J))
for i in range(N):
    for j in range(J):
        U[i][j] = np.linalg.norm(X[i] - kmeans.cluster_centers_[j])

# Solve RBF model, predict and plot
w = np.dot((np.linalg.inv(np.dot(U.T, U))), U.T) * y
yh_rbf = np.dot(U, w)
plt.plot(y, yh_rbf, '.', color='red')
plt.xlabel('Target')
plt.ylabel('Predicted')
plt.title('RBF model')
plt.show()
print(np.linalg.norm(y - yh_lin), np.linalg.norm(y - yh_rbf))
