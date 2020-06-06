import numpy as np


def hamming_distance(X, X_train):
    """
    Zwróć odległość Hamminga dla obiektów ze zbioru *X* od obiektów z *X_train*.

    :param X: zbiór porównywanych obiektów N1xD
    :param X_train: zbiór obiektów do których porównujemy N2xD
    :return: macierz odległości pomiędzy obiektami z "X" i "X_train" N1xN2
    """
    # X = X.astype(int)
    # X_train = X_train.astype(int)
    X = X / 255.
    X_train = X_train / 255.
    distance = np.empty((len(X), len(X_train)))
    for i in range(np.shape(X)[0]):
        distance[i] = np.sum(abs(X_train - X[i]), axis=1)
        if i % 100 == 0:
            print(i/np.shape(X)[0], "%")
    return distance


def sort_train_labels_knn(Dist, y):
    """
    Posortuj etykiety klas danych treningowych *y* względem prawdopodobieństw
    zawartych w macierzy *Dist*.

    :param Dist: macierz odległości pomiędzy obiektami z "X" i "X_train" N1xN2
    :param y: wektor etykiet o długości N2
    :return: macierz etykiet klas posortowana względem wartości podobieństw
        odpowiadającego wiersza macierzy Dist N1xN2

    Do sortowania użyj algorytmu mergesort.
    """
    indexes = np.argsort(Dist, kind='mergesort')
    return y[indexes]


def p_y_x_knn(y, k):
    """
    Wyznacz rozkład prawdopodobieństwa p(y|x) każdej z klas dla obiektów
    ze zbioru testowego wykorzystując klasyfikator KNN wyuczony na danych
    treningowych.

    :param y: macierz posortowanych etykiet dla danych treningowych N1xN2
    :param k: liczba najbliższych sasiadow dla KNN
    :return: macierz prawdopodobieństw p(y|x) dla obiektów z "X" N1xM
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
    Wyznacz błąd klasyfikacji.

    :param p_y_x: macierz przewidywanych prawdopodobieństw - każdy wiersz
        macierzy reprezentuje rozkład p(y|x) NxM
    :param y_true: zbiór rzeczywistych etykiet klas 1xN
    :return: błąd klasyfikacji
    """
    p_y_x_flipped = np.fliplr(p_y_x)
    args_max = len(p_y_x[0]) - 1 - np.argmax(p_y_x_flipped, axis=1)
    return np.mean(y_true != args_max)


def model_selection_knn(X_val, X_train, y_val, y_train, k_values):
    """
    Wylicz bład dla różnych wartości *k*. Dokonaj selekcji modelu KNN
    wyznaczając najlepszą wartość *k*, tj. taką, dla której wartość błędu jest
    najniższa.

    :param X_val: zbiór danych walidacyjnych N1xD
    :param X_train: zbiór danych treningowych N2xD
    :param y_val: etykiety klas dla danych walidacyjnych 1xN1
    :param y_train: etykiety klas dla danych treningowych 1xN2
    :param k_values: wartości parametru k, które mają zostać sprawdzone
    :return: krotka (best_error, best_k, errors), gdzie "best_error" to
        najniższy osiągnięty błąd, "best_k" to "k" dla którego błąd był
        najniższy, a "errors" - lista wartości błędów dla kolejnych
        "k" z "k_values"
    """
    best_err = 0
    best_k = 0
    errors = []

    distance_matrix = hamming_distance(X_val, X_train)
    sorted_labels = sort_train_labels_knn(distance_matrix, y_train)

    for k in k_values:
        p_y_x = p_y_x_knn(sorted_labels, k)
        class_err = classification_error(p_y_x, y_val)
        errors.append(class_err)
        if class_err == min(errors):
            best_err = class_err
            best_k = k
    return (best_err, best_k, errors)
