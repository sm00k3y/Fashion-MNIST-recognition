from utils import mnist_reader
import numpy as np

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

k = 11  # k do testów
train_count = np.shape(X_train)[0]
test_count = np.shape(X_test)[0]

predictions = np.empty(train_count)
err_number = 0

for i_test_img in range(test_count):  # 3 for now, test_size for later

    # test_img_distances = np.array([])
    # test_img_distances = np.empty(train_count)

    # test_img_distances = [np.sum(((X_test[i_test_img] - X_train[j])**2)**(0.5)) for j in range(train_count)]
    test_img_distances = [np.sum(abs(X_test[i_test_img] - X_train[j])) for j in range(train_count)]

    # for j_train_img in range(train_count):
    #     distance = np.sum(((X_test[i_test_img] - X_train[j_train_img])**2)**(0.5))
    #     # test_img_distances = np.append(test_img_distances, distance)
    #     test_img_distances[j_train_img] = distance

    sorted_indexes = np.argsort(test_img_distances)
    k_labels = y_train[sorted_indexes[0:k]]
    values, counts = np.unique(k_labels, return_counts=True)

    # predictions = np.append(predictions, values[np.argmax(counts)])
    predictions[i_test_img] = values[np.argmax(counts)]

    if predictions[i_test_img] != y_test[i_test_img]:
        err_number += 1
        print(err_number, "/", i_test_img)
    print(i_test_img, end="\t")
    print("# Classification Errors = ", err_number, "\t accuracy = ", 100.*(test_count - err_number) / test_count, "%")
