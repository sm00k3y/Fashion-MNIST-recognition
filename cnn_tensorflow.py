import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import categorical_crossentropy

BATCH_SIZE = 32         # Size of one batch
NUM_OF_EPOCHS = 25      # Number of epoch to train the model
VAL_SPLIT = 0.2         # Validation set split ratio (20% of dataset will be validation)
LEARNING_RATE = 0.01    # Learning rate for Stochastic Gradient Descent
LABELS_COUNT = 10


def prepare_dataset(x_train, y_train, x_test, y_test):
    """
    Preparing dataset: reshaping, normalizing, categorizing, splitting
    """
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = x_train / 255.
    x_test = x_test / 255.

    y_train = to_categorical(y_train, LABELS_COUNT)
    y_test = to_categorical(y_test, LABELS_COUNT)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=VAL_SPLIT, random_state=2018)

    return x_train, y_train, x_val, y_val, x_test, y_test


def cnn_model():
    """
    Building a Sequential model, displaying its statistics at the end
    """
    # First set of 2 * Convolutional layers
    model = Sequential()
    model.add(Conv2D(32,
                     kernel_size=(3, 3),
                     activation='relu',
                     padding='same',
                     kernel_initializer='he_normal',
                     input_shape=(28, 28, 1)))
    model.add(BatchNormalization(axis=-1))
    model.add(Conv2D(32,
                     kernel_size=(3, 3),
                     activation='relu',
                     padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Second set of 2 * Convolutional layers
    model.add(Conv2D(64,
                     kernel_size=(3, 3),
                     activation='relu',
                     padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(Conv2D(64,
                     kernel_size=(3, 3),
                     activation='relu',
                     padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # One normal, dense hidden layer with size of 512
    model.add(Flatten())
    model.add(Dense(512,
                    activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # Last layer with 10 neurons representing labels
    model.add(Dense(LABELS_COUNT,
                    activation='relu'))

    opt = SGD(lr=LEARNING_RATE, momentum=0.9, decay=LEARNING_RATE/NUM_OF_EPOCHS)

    model.compile(loss=categorical_crossentropy,
                  optimizer=opt,
                  metrics=['accuracy'])

    model.summary()

    return model


def run_model(model, X_train, y_train, X_val, y_val, X_test, y_test, class_names):
    """
    Training the model with parameters at the beginning, then displaying
    test score and classification report
    """
    train_model = model.fit(X_train, y_train,
                            batch_size=BATCH_SIZE,
                            epochs=NUM_OF_EPOCHS,
                            verbose=1,
                            validation_data=(X_val, y_val))
    print()
    predictions = model.predict(X_test)
    print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1), target_names=class_names))

    score = model.evaluate(X_test, y_test, verbose=1)
    print("\nTest loss: {:.3f}".format(score[0]))
    print("Test accuracy: {:.3f}\n".format(score[1]))

    return train_model


def plot_model_evaluation(train_model):
    """
    Plotting some results from training the model
    """
    history = train_model.history

    train_acc = history['accuracy']
    val_acc = history['val_accuracy']

    plt.style.use("dark_background")

    plt.figure()
    plt.plot(range(1, NUM_OF_EPOCHS+1), train_acc)
    plt.plot(range(1, NUM_OF_EPOCHS+1), val_acc)
    plt.xlabel("Number of epochs")
    plt.ylabel("Accuracy")
    plt.legend(["Training accuracy", "Validation accuracy"])
    plt.title("Accuracy of training and validation dataset")
    plt.draw()
    plt.waitforbuttonpress()

    train_loss = history['loss']
    val_loss = history['val_loss']

    plt.figure()
    plt.plot(range(1, NUM_OF_EPOCHS+1), train_loss)
    plt.plot(range(1, NUM_OF_EPOCHS+1), val_loss)
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss")
    plt.legend(["Training loss", "Validation loss"])
    plt.title("Loss of training and validation dataset")
    plt.draw()
    plt.waitforbuttonpress()
