from utils import mnist_reader
import matplotlib.pyplot as plt
import numpy as np
import time
import knn
import cnn_tensorflow as cnn

# Importing data from local files
X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print("---------------------------------------------------------------------")
print("| Dimensions of Train Set")
print("| Dimension(X_train)=", np.shape(X_train))
print("| There are", np.shape(X_train)[0], "images where each image is", np.shape(X_train)[1:], "pixels in size")
print("| There are", np.shape(np.unique(y_train))[0], "unique image labels")
print("---------------------------------------------------------------------")
print("| Dimensions of Test Set")
print("| Dimension(X_test)=", np.shape(X_test), "Dimension(y_test)=", np.shape(y_test)[0])
print("---------------------------------------------------------------------")


def show_sample_dataset():
    """
    Showing first 40 images of train dataset
    """
    xTrain = np.reshape(X_train, (np.shape(X_train)[0], 28, 28))
    plt.figure(figsize=(14, 10))
    plt.style.use('default')

    for i in range(40):
        plt.subplot(5, 8, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(xTrain[i], cmap=plt.cm.gist_yarg)
        plt.xlabel(class_names[y_train[i]])

    plt.draw()
    plt.waitforbuttonpress()
    plt.close()


def knn_algorithm():

    start = time.time()

    # Running the KNN algorithm
    (best_err, best_k, errors) = knn.model_selection_knn(X_test, X_train, y_test, y_train, range(1, 15))

    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    # Presenting results
    print("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
    print("Best k: {}".format(best_k))
    print("Best (lowest) error: {}".format(best_err))
    print("Accuracy: {}".format(1 - best_err))
    print("Error table for each k: ", errors)
    print("\n\n------------------- CLOSE THE CHART TO CONTINUE--------------------")

    # Plotting the result
    knn.plot_knn_errors(errors)


def cnn_tensorflow():

    start = time.time()

    # Running the CNN Tensorflow algorithm
    cnn.show_sample_dataset(X_train, y_train)
    (xTrain, yTrain, xVal, yVal, xTest, yTest) = cnn.prepare_dataset(X_train, y_train, X_test, y_test)
    model = cnn.cnn_model()
    train_model = cnn.run_model(model, xTrain, yTrain, xVal, yVal, xTest, yTest, class_names)

    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
    print("\n\n------------------- PRESS ANY KEY TO CONTINUE--------------------")

    # Plotting results
    cnn.plot_model_evaluation(train_model)


if __name__ == "__main__":

    show_sample_dataset()

    print("\n---------------")
    print("| RUNNING KNN |")
    print("---------------")
    knn_algorithm()

    print("\n--------------------------")
    print("| RUNNING CNN TENSORFLOW |")
    print("--------------------------")
    cnn_tensorflow()
