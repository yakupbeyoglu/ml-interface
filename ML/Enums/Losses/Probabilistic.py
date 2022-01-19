from ..BaseEnum import BaseEnum


class Probabilistic(BaseEnum):
    # All Related information found on https://keras.io/api/losses/probabilistic_losses/

    # computes the cross-entropy loss between true labels and predicted labels
    # require y_true (either 0 or 1 ), and y_pred(predicted value)
    binary_crossentropy = 1

    # computes cross-entropy loss between labels and predictions
    # require y_true and y_pred(predicted value)
    categorical_crossentropy = 2

    # same as categorical but classes/labels with integer instead of class/label name
    # require y_true and y_pred(predicted value)
    sparse_categorical_crossentropy = 3

    # poisson loss between y_true and y_pred
    # loss = y_pred - y_true * log(y_pred)
    poisson = 4

    # binary crossentropy loss
    binary_crossentropy = 5

    # Kullback-Leibler divergence between y_true and y_pred
    # loss = y_true * log(y_true / y_pred)
    # https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
    kl_divergence = 6
