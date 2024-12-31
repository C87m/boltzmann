import matplotlib.pyplot as plt 
from sklearn.datasets import fetch_openml
X,y= fetch_openml(name='mnist_784', version=1,return_X_y=True)
print(X.data.shape)
plt.imshow(X[0].reshape(28,28), cmap=plt.cm.gray_r)
plt.show()
