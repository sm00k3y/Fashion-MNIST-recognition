from utils import mnist_reader
import matplotlib.pyplot as plt
import numpy as np
import time
import knn
import cnn_tensorflow as tf

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

# TESTING SPACE

# tf.show_sample_dataset(X_train, y_train)
# tf.cnn_model(X_train, y_train, X_test, y_test)
tf.run_test_harness(X_train, y_train, X_test, y_test)

# END OF TESTING SPACE


def knn_algorithm():

    start = time.time()

    (best_err, best_k, errors) = knn.model_selection_knn(X_test[0:200], X_train, y_test[0:200], y_train, range(1, 15))
    # (best_err, best_k, errors) = knn.model_selection_knn(X_test, X_train, y_test, y_train, range(1, 15))

    end = time.time()
    print("Time: {:.2f}".format(end - start))
    print("Best k: {}".format(best_k))
    print("Best (lowest) error: {}".format(best_err))
    print("Accuracy: {}".format(1 - best_err))
    print("Error table for each k: ", errors)

    plt.style.use('dark_background')
    plt.plot(range(1, 15), (1-np.array(errors)), 'g')
    plt.legend(('Accuracy', ''))
    plt.ylabel('Accuracy')
    plt.xlabel('K Nearest Neighbors')
    plt.show()
