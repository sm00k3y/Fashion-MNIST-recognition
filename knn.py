import matplotlib.pyplot as plt
import numpy as np


def taxicab_metric(X, X_train):
    """
    Method returns a taxicab distance, also known as Manhattan distance between two images
    """
    # Normalizing the data
    X = X / 255.
    X_train = X_train / 255.

    distance = np.empty((len(X), len(X_train)))

    for i in range(np.shape(X)[0]):
        distance[i] = np.sum(abs(X_train - X[i]), axis=1)
        if i % 100 == 0:
            print(100.*i/np.shape(X)[0], "%")   # Print progress every 1%

    return distance


def sort_train_labels_knn(Dist, y):
    """
    Sorting training labels *y* in regards to distances in matrix *Dist*
    """
    indexes = np.argsort(Dist, kind='mergesort')
    return y[indexes]


def p_y_x_knn(y, k):
    """
    Calculating probability distribution p(y|x) of every label for every object
    in test dataset using KNN
    """
    classes = np.unique(y[0])
    p_y_x = []

    for y_row in y:
        counts = np.array([np.count_nonzero(y_row[0:k] == k_class) for k_class in classes])
        row = counts / k
        p_y_x.append(row)

    return np.array(p_y_x)


def classification_error(p_y_x, y_true):
    """
    Calculating classification error
    """
    p_y_x_flipped = np.fliplr(p_y_x)
    args_max = len(p_y_x[0]) - 1 - np.argmax(p_y_x_flipped, axis=1)
    return np.mean(y_true != args_max)


def model_selection_knn(X_val, X_train, y_val, y_train, k_values):
    """
    Selecting the best *k* value for which value of classification error is the lowest

    Return:
    best_error: value of the lowest classification error
    best_k:     value of k for which error was the lowest
    errors:     list of error values for each k
    """
    best_err = 0
    best_k = 0
    errors = []

    distance_matrix = taxicab_metric(X_val, X_train)
    sorted_labels = sort_train_labels_knn(distance_matrix, y_train)

    for k in k_values:
        p_y_x = p_y_x_knn(sorted_labels, k)
        class_err = classification_error(p_y_x, y_val)
        errors.append(class_err)
        if class_err == min(errors):
            best_err = class_err
            best_k = k

    return (best_err, best_k, errors)


def plot_knn_errors(errors):
    """
    Plotting the errors of each *k* value
    """
    plt.figure()
    plt.style.use('dark_background')
    plt.plot(range(1, 15), (1-np.array(errors)), 'g')
    plt.legend(('Accuracy', ''))
    plt.ylabel('Accuracy')
    plt.xlabel('K Nearest Neighbors')
    plt.draw()
    # plt.show()
    plt.waitorbuttonpress()
    plt.close()
