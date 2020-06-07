from utils import mnist_reader
import numpy as np
import time
import knn
import cnn_tensorflow as cnn

# Importing data from local files
X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

print("---------------------------------------------------------------------")
print("| Dimensions of Train Set")
print("| Dimension(X_train)=", np.shape(X_train))
print("| There are", np.shape(X_train)[0], "images where each image is", np.shape(X_train)[1:], "pixels in size")
print("| There are", np.shape(np.unique(y_train))[0], "unique image labels")
print("---------------------------------------------------------------------")
print("| Dimensions of Test Set")
print("| Dimension(X_test)=", np.shape(X_test), "Dimension(y_test)=", np.shape(y_test)[0])
print("---------------------------------------------------------------------")


def knn_algorithm():

    start = time.time()

    # Running the KNN algorithm
    (best_err, best_k, errors) = knn.model_selection_knn(X_test, X_train, y_test, y_train, range(1, 15))

    end = time.time()
    # Presenting results
    print("Time: {:.2f}".format(end - start))
    print("Best k: {}".format(best_k))
    print("Best (lowest) error: {}".format(best_err))
    print("Accuracy: {}".format(1 - best_err))
    print("Error table for each k: ", errors)
    print("\n\n------------------- PRRESS ANY KEY TO CONTINUE--------------------")

    # Plotting the result
    knn.plot_knn_errors(errors)


def cnn_tensorflow():

    start = time.time()

    # Running the CNN Tensorflow algorithm
    cnn.show_sample_dataset(X_train, y_train)
    (xTrain, yTrain, xVal, yVal, xTest, yTest) = cnn.prepare_dataset(X_train, y_train, X_test, y_test)
    model = cnn.cnn_model()
    train_model = cnn.run_model(model, xTrain, yTrain, xVal, yVal, xTest, yTest)

    end = time.time()
    print("Time: {:.2f} seconds".format(end - start))
    print("\n\n------------------- PRRESS ANY KEY TO CONTINUE--------------------")

    # Plotting results
    cnn.plot_model_evaluation(train_model)


if __name__ == "__main__":
    knn_algorithm()
    cnn_tensorflow()
