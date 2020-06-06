from utils import mnist_reader
import numpy as np
import time
import knn

X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

print("--------------------------")
print("Dimensions of Train Set")
print("Dimension(X_train)=", np.shape(X_train))
print("There are", np.shape(X_train)[0], "images where each image is", np.shape(X_train)[1:], "in size")
print("There are", np.shape(np.unique(y_train))[0], "unique image labels")
print("--------------------------")
print("Dimensions of Test Set")
print("Dimension(X_test)=", np.shape(X_test), "Dimension(y_test)=", np.shape(y_test)[0])
print("--------------------------")

start = time.time()

(best_err, best_k, errors) = knn.model_selection_knn(X_test[0:100], X_train, y_test[0:100], y_train, range(1, 15))

end = time.time()
print("Time: {:.2f} seconds".format(end - start))
print("Best k: {}".format(best_k))
print("Best (lowest) error: {}".format(best_err))
print("Accuracy: {}".format(1 - best_err))
print("Error table for each k: ", errors)
